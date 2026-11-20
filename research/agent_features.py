"""Restricted, deterministic formula language; never eval/exec model output."""
import ast
import numpy as np


def parsed(expression,allowed):
    tree=ast.parse(expression,mode='eval')
    nodes=list(ast.walk(tree))
    if len(nodes)>80:raise ValueError('Expression too complex')
    functions={'pdiv':2,'logabs':1,'sqrtabs':1,'abs':1}
    names=set()
    def validate(n):
        if isinstance(n,ast.Expression):return validate(n.body)
        if isinstance(n,ast.Name):
            if n.id not in allowed:raise ValueError('Forbidden input: '+n.id)
            names.add(n.id);return
        if isinstance(n,ast.Constant) and type(n.value) in (int,float) and np.isfinite(n.value) and abs(n.value)<=100:return
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult)):
            validate(n.left);validate(n.right);return
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):validate(n.operand);return
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in functions and not n.keywords and len(n.args)==functions[n.func.id]:
            for arg in n.args:validate(arg)
            return
        raise ValueError('Unsupported formula syntax')
    validate(tree)
    if not names:raise ValueError('Feature must use a measured input')
    return tree,names


def evaluate(expression,frame):
    tree,_=parsed(expression,set(frame.columns))
    def f(n):
        if isinstance(n,ast.Expression):return f(n.body)
        if isinstance(n,ast.Name):return frame[n.id].to_numpy(float)
        if isinstance(n,ast.Constant):return float(n.value)
        if isinstance(n,ast.UnaryOp):return -f(n.operand) if isinstance(n.op,ast.USub) else f(n.operand)
        if isinstance(n,ast.BinOp):
            a,b=f(n.left),f(n.right)
            return a+b if isinstance(n.op,ast.Add) else a-b if isinstance(n.op,ast.Sub) else a*b
        a=[f(v) for v in n.args];name=n.func.id
        if name=='pdiv':return a[0]/np.where(np.abs(a[1])<1e-4,np.where(a[1]<0,-1e-4,1e-4),a[1])
        if name=='logabs':return np.log1p(np.abs(a[0]))
        if name=='sqrtabs':return np.sqrt(np.abs(a[0]))
        return np.abs(a[0])
    with np.errstate(all='ignore'):out=np.asarray(f(tree),float)
    out=np.broadcast_to(out,(len(frame),)).copy();out[~np.isfinite(out)]=np.nan
    return out


def augment(frame,columns,features):
    x=frame[columns].copy();added=[]
    for item in features:
        name=item['name']
        if not name.startswith('agent_') or name in x:raise ValueError('Invalid or duplicate feature name')
        _,deps=parsed(item['formula'],set(columns)|set(item.get('inputs',[])))
        if not deps.issubset(columns):continue
        x[name]=evaluate(item['formula'],frame[columns]);added.append(name)
    return x,added
