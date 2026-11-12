"""Validation aggregation for identical target coverage at each native context."""
import math
from research.general_rsi_context_length_evaluation_v1 import evaluate

def score(model,data,documents=None,context=None):
    length=model.spec.context if context is None else context
    indices=list(range(len(data.indexes['val']))) if documents is None else list(documents)
    assert len(indices)==len(set(indices)) and all(0<=i<len(data.indexes['val']) for i in indices)
    result=evaluate(model,data.view.val_data,[data.indexes['val'][i] for i in indices],length)
    return aggregate(result,indices,data.indexes['val'])

def aggregate(result,indices,documents):
    length=result['context'];docs=[]
    for local,index in enumerate(indices):
        rows=[r for r in result['rows'] if r['document']==local]
        docs.append(dict(document=index,source=documents[index]['source'],loss_sum=math.fsum(r['loss_sum'] for r in rows),count=sum(r['targets'] for r in rows)))
    for row in result['rows']:row['document']=indices[row['document']]
    sources={}
    for source in sorted({d['source'] for d in docs}):
        selected=[d for d in docs if d['source']==source];count=sum(d['count'] for d in selected);loss=math.fsum(d['loss_sum'] for d in selected)
        sources[source]=dict(count=count,loss_sum=loss,ce=loss/count)
    count=sum(d['count'] for d in docs);loss=math.fsum(d['loss_sum'] for d in docs)
    return dict(context=length,count=count,loss_sum=loss,ce=loss/count,sources=sources,documents=docs,chunks=result['rows'])
