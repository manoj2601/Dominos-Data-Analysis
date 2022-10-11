# import Pkg; Pkg.add("OpenStreetMap")
using OpenStreetMapX

############################################### Change This Only ###########################################
#output file for storing nodes
outfile1 = open("tempnodes.csv", "w")
outfile2 = open("tempsegments.csv", "w")
println("manoj start")
md = get_map_data("/media/root/data/analysis/finalMap.xml", trim_to_connected_graph=false, only_intersections=true)
println("manoj end")
exit(1)

# File 1: 
# node_id, lat, long
# FILE 2:
# node1, node2, dist (if possible time)
