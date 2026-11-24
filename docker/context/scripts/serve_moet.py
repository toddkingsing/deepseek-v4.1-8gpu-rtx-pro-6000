"""Own a validated Docker model; eight-GPU deployments retry without silently shrinking."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
ROOT=Path(os.environ.get('DS_ROOT','/opt/deepseek-deploy'))
STAMP=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
KIND=os.environ.get('DS_MODEL_KIND','official')
GPU_DEVICES=[x.strip() for x in os.environ.get('DS_GPU_DEVICES','0,1,2,3').split(',') if x.strip()]
GPU_CSV=','.join(GPU_DEVICES)
REQUIRE_MOET_ONLY=os.environ.get('DS_RUNTIME')=='moet' and set(GPU_DEVICES) in ({'0','1','2','3'},{'0','1','2','3','4','5','6','7'})
assert KIND in {'official'}
NAME='deepseek-'+KIND+'-serving-'+STAMP.lower()
ACTIVE=ROOT/'results/active-runtime.json'
EVENT=ROOT/'results'/f'serving-runtime-{STAMP}.json'
record={'started':time.time(),'launcher_pid':os.getpid(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'mode':'starting','container':NAME,'model_kind':KIND}
stopping=False;created=False;log_proc=None;budget_ok=True

def save(mode,**values):
 record.update(mode=mode,**values)
 data=json.dumps(record,indent=2)+'\n';temp=ACTIVE.with_suffix('.tmp');temp.write_text(data);temp.replace(ACTIVE);EVENT.write_text(data)
 print(json.dumps({'runtime':mode,'container':NAME,**values}),flush=True)

def docker(*args,check=True,timeout=15):
 return subprocess.run(['sudo','-n','docker',*args],capture_output=True,text=True,check=check,timeout=timeout)

def stop(sig,frame):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

def stop_owned_container(name):
 last_error=None
 for attempt in range(2):
  try:
   docker('stop','-t','60',name,check=False,timeout=100)
   state=json.loads(docker('inspect','--format','{{json .State}}',name).stdout)
   if not state['Running']:return state
   last_error=RuntimeError('Owned model container still running')
  except (subprocess.SubprocessError,ValueError) as error:last_error=error
  time.sleep(3)
 raise RuntimeError('Cannot recover while the owned Docker model may still hold GPUs') from last_error

def cleanup():
 global log_proc
 try:
  if created:record['container_exit']=stop_owned_container(NAME)
 finally:
  if log_proc:
   log_proc.terminate()
   try:log_proc.wait(timeout=10)
   except subprocess.TimeoutExpired:log_proc.kill();log_proc.wait()
   log_proc=None

def recover_orphan():
 if not ACTIVE.exists():return
 old=json.loads(ACTIVE.read_text());name=old.get('container');pid=old.get('launcher_pid')
 if not name or name==NAME or old.get('model_kind')!=KIND:return
 # Only a prior launcher recorded by this service, whose process is gone.
 if not isinstance(pid,int) or Path('/proc',str(pid)).exists():return
 own=docker('inspect','--format','{{index .Config.Labels "deepseek.task"}}',name,check=False)
 if own.returncode or own.stdout.strip()!=KIND+'-serving':return
 state=json.loads(docker('inspect','--format','{{json .State}}',name).stdout)
 if state['Running']:
  record['orphan_recovery']={'container':name,'state':stop_owned_container(name)}

def failure_budget(record_failure=False):
 path=ROOT/'results/eight-recovery-budget.json';now=time.time()
 old=json.loads(path.read_text()) if path.exists() else {}
 history=[t for t in old.get('failed_launches',[]) if now-21600<t<=now]
 if record_failure:
  history.append(now);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps({'failed_launches':history,'recorded':now})+'\n');tmp.replace(path)
 return len(history)<3

try:
 assert KIND in {'official'}
 if REQUIRE_MOET_ONLY:
  budget_ok=failure_budget()
  assert budget_ok,'Three failed launches in six hours; inspect eight-recovery-budget.json before retrying'
 assert os.environ.get('NCCL_P2P_DISABLE')=='0','Peer check failed; use native recovery'
 manifest=json.loads((ROOT/('config/moet-serving.json' if KIND=='official' else 'config/moet-serving-official.json')).read_text())
 source=Path(manifest['source_trial']);assert source.parent==ROOT/'results'
 assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_sha256']
 deadline=time.monotonic()+120
 while docker('info',check=False,timeout=15).returncode:
  if stopping:raise KeyboardInterrupt
  if time.monotonic()>deadline:raise RuntimeError('Docker daemon did not become ready')
  time.sleep(3)
 assert docker('image','inspect','--format','{{.Id}}',manifest['image_id']).stdout.strip()==manifest['image_id']
 recover_orphan()
 assert not subprocess.check_output(['nvidia-smi','-i',GPU_CSV,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU processes already running'
 cmd=list(manifest['command_template']);assert cmd[:4]==['sudo','-n','docker','run']
 cmd[cmd.index('--name')+1]=NAME
 cmd[cmd.index('--label')+1]='deepseek.task='+KIND+'-serving'
 image_index=cmd.index('--entrypoint')+2;cmd[image_index]=manifest['image_id']
 settings={'image_id':manifest['image_id'],'env':[cmd[i+1] for i,x in enumerate(cmd) if x=='-e'],'args':cmd[image_index+1:]}
 fingerprint=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
 assert fingerprint==manifest['inference_fingerprint'],'Serving parameters differ from the validated trial'
 for variable,flag in [('DS_CONTEXT','--max-model-len'),('DS_SEQUENCES','--max-num-seqs'),('DS_BATCH','--max-num-batched-tokens'),('DS_KV_BYTES','--kv-cache-memory-bytes'),('DS_LINEAR','--linear-backend')]:
  assert os.environ.get(variable)==cmd[cmd.index(flag)+1],f'{variable} differs from the validated Moet profile'
 spec=json.loads(cmd[cmd.index('--speculative-config')+1])
 assert int(os.environ['DS_DRAFT'])==spec['num_speculative_tokens'],'DS_DRAFT differs from the validated Moet profile'
 runtime_env=dict(x.split('=',1) for x in settings['env'])
 if REQUIRE_MOET_ONLY:
  tp_sz=int(cmd[cmd.index('--tensor-parallel-size')+1])
  pp_sz=int(cmd[cmd.index('--pipeline-parallel-size')+1]) if '--pipeline-parallel-size' in cmd else 1
  assert tp_sz*pp_sz==len(GPU_DEVICES),f'Moet profile worker count {tp_sz*pp_sz} != {len(GPU_DEVICES)}'
  if len(GPU_DEVICES)==8:
   assert cmd[cmd.index('--gpus')+1]=='all','Eight-GPU profile requires all Docker GPUs'
   assert runtime_env.get('NVIDIA_VISIBLE_DEVICES','all')=='all','Eight-GPU profile cannot hide NVIDIA devices'
   assert set(runtime_env.get('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7').split(','))==set(map(str,range(8))),'Eight-GPU profile requires all eight visible devices'
  else:
   assert cmd[cmd.index('--gpus')+1]==f'"device={GPU_CSV}"',f'Four-GPU profile requires isolated --gpus "device={GPU_CSV}"'
 assert os.environ.get('DS_MOE_A16','1')==runtime_env.get('VLLM_B12X_MOE_FP4_FORCE_A16','1'),'MoE activation precision differs from validated profile'
 assert (os.environ.get('DS_ADAPTIVE','false').lower()=='true')==spec.get('enable_adaptive_verification',False),'Adaptive verification differs from validated profile'
 assert os.environ.get('PYTORCH_ALLOC_CONF')==runtime_env['PYTORCH_ALLOC_CONF'],'Allocator differs from the validated Moet profile'
 profiles=ROOT/'profiles'/'serving'/STAMP;profiles.mkdir(parents=True,exist_ok=True)
 for i,item in enumerate(cmd):
  if item.startswith('type=bind,src=') and item.endswith(',dst=/profiles'):cmd[i]=f'type=bind,src={profiles},dst=/profiles'
 if stopping:raise KeyboardInterrupt
 save('launching',source_trial=manifest['source_trial'],image_id=manifest['image_id'],inference_fingerprint=fingerprint)
 container_id=subprocess.check_output(cmd,text=True,timeout=90).strip();created=True
 state=json.loads(docker('inspect','--format','{{json .State}}',NAME).stdout);assert state['Running']
 # Docker's raw cgroup limit can be reset by systemd daemon-reload unless
 # its transient scope carries the same policy. Enforce before serving.
 group=Path('/proc',str(state['Pid']),'cgroup').read_text().strip().split(':')[-1]
 scope='docker-'+container_id+'.scope'
 assert group=='/system.slice/'+scope,'Unexpected Docker cgroup'
 subprocess.run(['sudo','-n','systemctl','set-property','--runtime',scope,'MemorySwapMax=0'],check=True,timeout=30)
 assert Path('/sys/fs/cgroup'+group+'/memory.swap.max').read_text().strip()=='0','Docker swap policy not enforced'
 save('docker',container_id=container_id,swap_policy='systemd_scope_zero')
 log_proc=subprocess.Popen(['sudo','-n','docker','logs','--follow',NAME])
 inspect_failures=0
 while not stopping:
  try:
   state=json.loads(docker('inspect','--format','{{json .State}}',NAME).stdout)
  except (subprocess.SubprocessError,ValueError) as error:
   inspect_failures+=1
   print(json.dumps({'event':'docker_inspect_unavailable','consecutive':inspect_failures,'reason':repr(error)}),flush=True)
   if inspect_failures>=20:raise RuntimeError('Docker state unavailable after 20 consecutive probes') from error
   time.sleep(3);continue
  inspect_failures=0
  if not state['Running']:raise RuntimeError(f'Model container exited with code {state["ExitCode"]}')
  time.sleep(3)
 cleanup();save('stopped',finished=time.time());raise SystemExit(0)
except (Exception,KeyboardInterrupt) as error:
 if not created:
  try:
   ownership=docker('inspect','--format','{{index .Config.Labels "deepseek.task"}}',NAME,check=False)
   created=ownership.returncode==0 and ownership.stdout.strip()==KIND+'-serving'
  except Exception:pass
 cleanup()
 if stopping:
  save('stopped',finished=time.time());raise SystemExit(0)
 # A failed container must be gone before the known native service path is used.
 if subprocess.check_output(['nvidia-smi','-i',GPU_CSV,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
  save('failed',reason=repr(error),finished=time.time());raise SystemExit(1)
 if REQUIRE_MOET_ONLY:
  if budget_ok:failure_budget(record_failure=True)
  save('eight_gpu_retry',reason=repr(error),finished=time.time())
  # deepseek.service owns Restart=always. Never substitute the old four-GPU profile.
  raise SystemExit(1)
 save('native_fallback',reason=repr(error))
 fallback=locals().get('manifest',{}).get('native_fallback_profile')
 if fallback:
  fallback_path=Path(fallback).resolve()
  assert fallback_path.parent==(ROOT/'config/profiles').resolve(),'Fallback profile must be local'
  for line in fallback_path.read_text().splitlines():
   if '=' in line and not line.startswith('#'):
    key,value=line.split('=',1);os.environ[key]=value
 os.environ['DS_RUNTIME']='native'
 os.execv('/bin/bash',['bash',str(ROOT/'scripts/start_model.sh')])
