#maximum flow
"""
This is the max flow algorithm for ex8.31
Since all directed paths has 1 as min value flow, it will be exterminated after 1 time checking.
"""
startend=["s","t"]
node=["s","x1","x2","x3","x4","y1","y2","y3","y4","y5","y6","t"]
edge={"s":["x1","x2","x3","x4"],"x1":["y1","y2","y4","y5"],"x2":["y3","y6"],"x3":["y1","y3"],"x4":["y1","y3"],"y1":["t"],"y2":["t"],"y3":["t"],"y4":["t"],"y5":["t"],"y6":["t"]}
maxflow=0
fc=0
#check for directed paths if it connect s to t, the path that walked will be renamed "no" and made unavailable 
for b in edge["s"]:
    if b!="no":
        for c in edge[b]:
            if c!="no":
                for d in edge[c]:
                    if d=="t":
                        #add value to fc and turn path to unavailable
                        fc+=1
                        b="no"
                        c="no"
print(fc)






















