import os




with open("raw_index.rst","r") as f:
    base=f.read()

each=""".. automodule:: ####
    :members:
    :show-inheritance:
    :special-members:
"""
nnot="""

    :undoc-members:

"""

def iterpy(folder="xevo"):
    for f in os.listdir(folder):
        if os.path.isfile(folder+"/"+f):
            if not ".py" in f:continue
            if ".pyc" in f:continue
            if "__pycache__" in f:continue
            if "__init__" in f:continue
            yield folder+"/"+f
        else:
            for zx in iterpy(folder+"/"+f):
                yield zx



qst="\n\n".join([each.replace("####",zw.replace("/",".").replace(".py","")) for zw in iterpy()])
base=base.replace("####",qst)

with open("index.rst","w") as f:
    f.write(base)



if __name__=="__main__" and False:
    for zw in iterpy():
        print(zw)


