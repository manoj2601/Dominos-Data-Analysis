using OpenStreetMapX

city = "Bhopal"
PATH = "/media/root/data/analysis/map"

md = get_map_data("$(PATH)/OSMFiles/map_$(city).osm", use_cache=false, trim_to_connected_graph=true, only_intersections=false)

nodesf = open("$(PATH)/juliaGraph/nodes_$(city).csv", "w")
print(nodesf, "node_id,lat,long\n")

for i = 1:length(md.v)
    lat, long = OpenStreetMapX.latlon(md,i);
    println(nodesf,string(i)*","*string(lat)*","*string(long))
end

close(nodesf)

edgef = open("$(PATH)/juliaGraph/edges_$(city).csv", "w")
print(edgef, "node1,node2,dist\n")

inds = Tuple.(findall(!iszero, md.w))
for k in inds
    if(md.w[k] == 0.0)
        continue
    end
    println(edgef, string(k[1])*","*string(k[2])*","*string(md.w[k]))
end

close(edgef)