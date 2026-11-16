#!/usr/bin/env python3
"""Create a sanitized GitHub export without modifying experimental sources.
Default is a dry-run. --write copies into a separate destination.
"""
import argparse
import ast
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import stat

MODEL_EXT = {'.safetensors', '.bin', '.pt', '.pth', '.pkl', '.pickle', '.msgpack', '.gguf', '.ggml', '.onnx', '.h5', '.hdf5', '.ckpt', '.tflite'}
CACHE_DIRS = {'wandb', 'swanlog', 'swanlab_logs', 'unsloth_compiled_cache', '__pycache__', '.ipynb_checkpoints', '.cache', '.locks', 'cache', '.venv', 'venv', 'node_modules'}
EXCLUDE_DIRS = {'.git', '.github-upload', '.github-upload-review', '.aws', '.codex', '.agents'}
# Values are never printed or included in the generated reports.
ASSIGNMENT = re.compile(r'''(?P<prefix>\b(?P<name>[A-Za-z_]*(?:api_?key|access_?token|auth_?token|zilliz_token|hf_token|password|passwd|secret_?key|client_?secret|token))\b["']?\s*[:=]\s*)(?P<quote>["'])(?P<value>[^"'\r\n]{1,512})(?P=quote)''', re.I)
TOKEN = re.compile(r'(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16})(?![A-Za-z0-9_-])')
URL_PASSWORD = re.compile(r'(?P<prefix>(?:https?|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^/\s:@]+:)(?P<value>[^@\s/]+)(?P<suffix>@)',re.I)
PLACEHOLDERS = {'none', 'null', 'auto', 'default', 'token', 'password', 'passwd', 'api_key', 'your_api_key', 'your-api-key', 'your_token', 'your_password', 'your-password', 'your-api-key-here', 'your_access_token', 'your_zilliz_token', 'changeme', 'change_me', 'secret', 'xxx', 'xxxx', 'your_secret_key', 'your_dashscope_api_key'}

def placeholder(v):
    return v.lower() in PLACEHOLDERS or '__SET_VIA_ENVIRONMENT__' in v or any(z in v.lower() for z in ('${', 'os.getenv(', 'os.environ', '<your', '<redacted', 'your_', 'your-', 'replace_', 'replace-', '填入', '替换', '你的', 'your ', 'sk-xxx', 'sk-your', '*'))

def is_secret(name, value):
    if not value.strip() or placeholder(value): return False
    n=name.lower()
    if 'password' in n or 'passwd' in n or 'secret' in n: return True
    return len(value)>=12 and not any(ch.isspace() for ch in value)

def classify(rel, size, max_bytes):
    p=Path(rel); parts=p.parts; ext=p.suffix.lower(); name=p.name
    if parts[0]=='milvus':return 'milvus'
    if parts[0] in ('bert-base-chinese','bert-base-multilingual-cased'):return 'pretrained_model_resource'
    if any(z in ('model','models','blobs','snapshots') for z in parts):return 'model_directory'
    if any(z.startswith('checkpoint-') for z in parts):return 'training_checkpoint'
    if ext in MODEL_EXT:return 'model_weight_or_serialized_detector'
    if any(z in CACHE_DIRS for z in parts):return 'generated_cache_or_tracking'
    if any(z in EXCLUDE_DIRS for z in parts):return 'local_configuration_or_export'
    if name=='.env' or name.startswith('.env.') and name not in ('.env.example','.env.sample') or ext in ('.pem','.key','.p12','.pfx'):return 'local_secret_configuration'
    if ext in ('.deb','.whl','.zip','.gz','.xz','.tar','.7z','.rar'):return 'installer_or_archive'
    if ext in ('.log','.wandb','.swanlab','.pyc','.lock') or name.startswith('tmp') or name in ('.DS_Store',):return 'log_or_temporary'
    if size >= max_bytes:return 'large_non_model_data'
    return None

def text_data(path):
    raw=path.read_bytes()
    if b'\x00' in raw[:8192]:return None
    try:return raw.decode('utf-8-sig')
    except UnicodeDecodeError:return None

def possible_assignments(text):
    lower=text.lower()
    markers=('api_key','apikey','api-key','password','passwd','secret','access_token','auth_token','zilliz_token','hf_token','"token"',"'token'",'token =','token=','_token"',"_token'")
    return any(marker in lower for marker in markers)

def possible_tokens(text):
    return any(marker in text for marker in ('sk-','ghp_','gho_','ghu_','ghs_','ghr_','github_pat_','hf_','AKIA'))

def window_matches(text, pattern, markers, lower=False):
    search=text.lower() if lower else text
    intervals=[]
    for marker in markers:
        start=0
        while True:
            pos=search.find(marker,start)
            if pos<0:break
            intervals.append((max(0,pos-100),min(len(text),pos+800)))
            start=pos+len(marker)
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    for start,end in merged:
        yield from pattern.finditer(text,start,end)

def assignment_matches(text):
    markers=('api_key','apikey','api-key','password','passwd','secret','access_token','auth_token','zilliz_token','hf_token','"token"',"'token'",'token =','token=','_token"',"_token'")
    return window_matches(text,ASSIGNMENT,markers,lower=True)

def token_matches(text):
    return window_matches(text,TOKEN,('sk-','ghp_','gho_','ghu_','ghs_','ghr_','github_pat_','hf_','AKIA'))

def url_matches(text):
    return window_matches(text,URL_PASSWORD,('://',)) if '@' in text else iter(())

def replace_matches(text,matches,fn):
    chunks=[];previous=0
    for match in matches:
        chunks.append(text[previous:match.start()]);chunks.append(fn(match));previous=match.end()
    if not chunks:return text
    chunks.append(text[previous:]);return ''.join(chunks)

def sanitize(text, known_secrets):
    counts=collections.Counter()
    def sub(m):
        v=m['value']
        if not is_secret(m['name'],v):return m[0]
        counts['credential_assignment']+=1
        return m['prefix']+m['quote']+'__SET_VIA_ENVIRONMENT__'+m['quote']
    text=replace_matches(text,assignment_matches(text),sub)
    def token(m):counts['token_pattern']+=1;return '__SET_VIA_ENVIRONMENT__'
    text=replace_matches(text,token_matches(text),token)
    def url(m):counts['url_password']+=1;return m['prefix']+'__SET_VIA_ENVIRONMENT__'+m['suffix']
    text=replace_matches(text,url_matches(text),url)
    for secret in known_secrets:
        if secret in text:
            counts['repeated_credential_value']+=text.count(secret)
            text=text.replace(secret,'__SET_VIA_ENVIRONMENT__')
    return text,dict(counts)

def prepare(args):
    root=Path(args.source).resolve();dest=Path(args.destination).resolve();review=Path(args.review).resolve()
    if dest==root or root.is_relative_to(dest):raise SystemExit('Destination must be separate from source.')
    review.mkdir(parents=True,exist_ok=True)
    included=[];excluded=[]
    for directory,ds,fs in os.walk(root,followlinks=False):
        ds[:]=[n for n in ds if n not in ('.git','.github-upload','.github-upload-review')]
        for name in fs:
            path=Path(directory)/name;rel=path.relative_to(root).as_posix()
            try:s=path.lstat()
            except OSError:continue
            row={'path':rel,'bytes':s.st_size}
            if stat.S_ISLNK(s.st_mode):reason='symlink_not_followed'
            elif not stat.S_ISREG(s.st_mode):reason='non_regular_file'
            else:reason=classify(rel,s.st_size,int(args.max_file_mib*1024**2))
            if reason:excluded.append({**row,'reason':reason})
            else:included.append(row)
    # Discover reusable secret strings, without displaying or persisting their values.
    secrets=set();sensitive_files=[];invalid_notebooks=[]
    for i,row in enumerate(included):
        p=root/row['path'];ext=p.suffix.lower()
        if ext not in ('.py','.sh','.json','.jsonl','.ipynb','.md','.txt','.yaml','.yml','.toml','.ini','.cfg','.conf','.env','.sql','.js','.ts'):continue
        text=text_data(p)
        if text is None:continue
        views=[text]
        if ext=='.ipynb':
            try:
                nb=json.loads(text);views.extend(''.join(c.get('source',[])) for c in nb.get('cells',[]))
            except (ValueError,TypeError):invalid_notebooks.append(row['path'])
        categories=collections.Counter()
        for view in views:
            for m in assignment_matches(view):
                if is_secret(m['name'],m['value']):
                    categories['credential_assignment']+=1
                    if len(m['value'])>=16:secrets.add(m['value'])
            for m in token_matches(view):
                if not placeholder(m[0]):secrets.add(m[0]);categories['token_pattern']+=1
            for m in url_matches(view):
                categories['url_password']+=1
                if len(m['value'])>=16:secrets.add(m['value'])
        if categories:sensitive_files.append({'path':row['path'],'findings':dict(categories)})
        if i and i%500==0:print('scanned',i,'of',len(included),flush=True)
    invalid=set(invalid_notebooks)
    kept=[]
    for row in included:
        if row['path'] in invalid:excluded.append({**row,'reason':'invalid_notebook_pending_review'})
        else:kept.append(row)
    included=kept
    plan={'source':str(root),'destination':str(dest),'max_file_mib':args.max_file_mib,'candidate_files':len(included),'candidate_bytes':sum(r['bytes'] for r in included),'excluded_files':len(excluded),'exclusion_counts':dict(collections.Counter(r['reason'] for r in excluded)),'files_with_potential_credentials':len(sensitive_files),'invalid_notebooks_excluded':invalid_notebooks,'mode':'written' if args.write else 'dry-run'}
    for name,rows,fields in [('included_files.csv',included,['path','bytes']),('excluded_files.csv',excluded,['path','bytes','reason'])]:
        with (review/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted(rows,key=lambda r:r['path']))
    (review/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    (review/'sensitive_paths.json').write_text(json.dumps(sensitive_files,ensure_ascii=False,indent=2)+'\n')
    if args.write:
        if dest.exists() and any(p.name!='.git' for p in dest.iterdir()):raise SystemExit('Destination is nonempty; refusing to overwrite existing files.')
        dest.mkdir(parents=True,exist_ok=True);changes=[];syntax_issues=[];output_bytes=0
        for i,row in enumerate(included):
            src=root/row['path'];dst=dest/row['path'];dst.parent.mkdir(parents=True,exist_ok=True)
            text=text_data(src);change={}
            if text is None:
                data=src.read_bytes();dst.write_bytes(data)
            elif src.suffix=='.ipynb':
                nb=json.loads(text);count=0;cnt=collections.Counter()
                for cell in nb.get('cells',[]):
                    old=''.join(cell.get('source',[]));new,cats=sanitize(old,secrets);cnt.update(cats);cell['source']=new.splitlines(keepends=True)
                    if cell.get('cell_type')=='code':count+=len(cell.get('outputs',[]));cell['outputs']=[];cell['execution_count']=None
                # Notebook metadata may contain execution paths and credentials too.
                nb=json.loads(sanitize(json.dumps(nb,ensure_ascii=False),secrets)[0]);dst.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
                change={'redactions':dict(cnt),'cleared_outputs':count}
            else:
                new,cats=sanitize(text,secrets);dst.write_text(new);change={'redactions':cats} if cats else {}
                if src.suffix=='.py' and new!=text:
                    try:ast.parse(text)
                    except SyntaxError:pass
                    else:
                        try:ast.parse(new)
                        except SyntaxError:syntax_issues.append(row['path'])
            if change and (change.get('redactions') or change.get('cleared_outputs')):changes.append({'path':row['path'],**change})
            dst.chmod(0o755 if src.stat().st_mode & stat.S_IXUSR else 0o644)
            output_bytes+=dst.stat().st_size
            if i and i%500==0:print('exported',i,'of',len(included),flush=True)
        info=dest/'github-export';info.mkdir(exist_ok=True)
        for name in ['included_files.csv','excluded_files.csv','plan.json']:(info/name).write_bytes((review/name).read_bytes())
        (review/'sanitization_report.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
        (info/'sanitization_report.json').write_bytes((review/'sanitization_report.json').read_bytes())
        plan['export_bytes']=output_bytes;plan['sanitized_files']=len(changes);plan['sanitization_syntax_issues']=syntax_issues
        (review/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');(info/'plan.json').write_bytes((review/'plan.json').read_bytes())
        if syntax_issues:raise SystemExit('Sanitization introduced Python syntax issues; review before committing.')
        # Independent scan of the sanitized copy against all discovered long secret values.
        residual=[]
        for row in included:
            text=text_data(dest/row['path'])
            if text is not None and (any(s in text for s in secrets) or any(not placeholder(m[0]) for m in token_matches(text))):residual.append(row['path'])
        (review/'residual_secret_paths.json').write_text(json.dumps(residual,ensure_ascii=False,indent=2)+'\n')
        if residual:raise SystemExit('Residual credential candidates remain; review sanitized export before committing.')
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',default='/mnt/project');ap.add_argument('--destination',default='/mnt/project/.github-upload/paper_test');ap.add_argument('--review',default='/mnt/project/.github-upload-review');ap.add_argument('--max-file-mib',type=float,default=50);ap.add_argument('--write',action='store_true');prepare(ap.parse_args())
