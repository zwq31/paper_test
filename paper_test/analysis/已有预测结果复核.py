import json,collections,math,os
from pathlib import Path
root=Path('/mnt/project');d=root/'H-Neurous/data/test'
truth=json.load(open(d/'test_inference_chat/evaluation_results.json'))
key=lambda z:(str(z.get('source_id','')),z.get('question','').strip())
t={key(z):z for z in truth}
def auc(y,p):
 a=sorted(zip(p,y));pos=sum(y);neg=len(y)-pos
 if not pos or not neg:return None
 rank=0;i=0
 while i<len(a):
  j=i+1
  while j<len(a) and a[j][0]==a[i][0]:j+=1
  rank+=sum(z[1] for z in a[i:j])*((i+1+j)/2);i=j
 return (rank-pos*(pos+1)/2)/(pos*neg)
def metrics(tp,fp,tn,fn):
 n=tp+fp+tn+fn;a=tp+fp
 return {'n':n,'tp_correct_kept':tp,'fp_wrong_kept':fp,'tn_wrong_rejected':tn,'fn_correct_rejected':fn,'coverage':a/n,'answer_accuracy':tp/a if a else None,'correct_all':tp/n,'decision_accuracy':(tp+tn)/n,'wrong_all':fp/n}
out={}
for f in sorted(d.glob('test_inference*.json')):
 data=json.load(open(f));counts=collections.Counter(); y=[];p=[]
 for z in data:
  v=t.get(key(z))
  if v is None:counts['unmatched']+=1;continue
  if z.get('model_answer','').strip()!=v.get('model_answer','').strip():counts['different_answer']+=1;continue
  score=v.get('evaluation_results',{}).get('correctness',{}).get('score');h=z.get('is_hallucination')
  if score is None or not isinstance(h,bool):counts['missing']+=1;continue
  good=score>=4;keep=not h;counts[('tp' if good else 'fp') if keep else ('fn' if good else 'tn')]+=1
  prob=z.get('hallucination_prob')
  if isinstance(prob,(int,float)) and math.isfinite(prob):y.append(int(not good));p.append(prob)
 n=sum(counts[v] for v in ['tp','fp','tn','fn'])
 if n:
  v=metrics(*[counts[k] for k in ['tp','fp','tn','fn']]);v.update({'input_n':len(data),'different_answer':counts['different_answer'],'unmatched':counts['unmatched'],'auc_hallucination':auc(y,p),'auc_n':len(y)});out[str(f.relative_to(root))]=v;print(f.name,json.dumps(v,ensure_ascii=False))
json.dump(out,open('/tmp/hneurons_external_recomputed.json','w'),ensure_ascii=False,indent=2)
# Agentic diagnostics only: no correctness inferred.
p=root/'Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/agentic_rag_system/data/test/test.json';a=json.load(open(p)); counts=collections.Counter()
for z in a:
 meta=z.get('process_metadata',{});v=meta.get('verification',{})
 if isinstance(v,dict):
  if v.get('is_supported') is True:counts['verification_supported']+=1
  if '验证过程异常' in str(v) or 'Invalid json output' in str(v):counts['verification_parse_exception']+=1
 if z.get('confidence')==0:counts['confidence_zero']+=1
print('AGENTIC',len(a),dict(counts));json.dump({'n':len(a),**counts},open('/tmp/agentic_diagnostics.json','w'),ensure_ascii=False)
# TriviaQA ground truth integrity
for name in ['triviaqa_rc_wikipedia_test_evaluate.json','triviaqa_rc_wikipedia_train_evaluate.json']:
 a=json.load(open(root/'unknow_test/output'/name));c=collections.Counter()
 for z in a:
  g=z.get('ground_truth',{});c['correct']+=int(z.get('correctness',{}).get('correct',False));c['unk']+=int(g.get('value')=='<unk>');c['empty_aliases']+=int(not g.get('aliases'))
 print('TRIVIA',name,len(a),dict(c))
