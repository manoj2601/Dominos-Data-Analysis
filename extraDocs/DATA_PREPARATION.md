# Environment 

Preprocessing scripts are written in Python3(Tested on Python 3.6.9)
Algorithm/Simulation codebase is in C++11 (Tested on GCC 7.4.0)

Create environment venv :
	conda env create -f environment.yml


################## Replace BLR with AHM/DEL accordingly ##################

## Swiggy Data
Swiggy's provided food related data resides in data_BLR/swiggy_data/ and delivery executive ping in data_BLR/DE_data/
Use scripts/truncate_swiggy_data.py to remove attributes not important for us.
The data in data_BLR/swiggy_data/ is assumed to be tab separated. Use scripts/csv_to_tsv.py to modify csv files. The generated files are csv.

Current Location : data_BLR/required_data/

## Getting Map
	Use scripts/get_osm_map.py to download a OSM using a given bounding box. It creates a commandline query to dowload the map.The bounding box can be found by experimenting with zones and orders. Use scripts/get_bounding_box.ipynb and http://bboxfinder.com/ for the same
	python3 scripts/get_osm_map.py
	Map File : data_BLR/map/map.xml

## Generating nodes and segments
	Use scripts/map_extract/finalParseBound.jl to extract nodes and edges from the OSM XML format map. It creates two files - nodes and segments

	julia scripts/map_extract/finalParseBound.jl

	Files Generated : data_BLR/map/nodes.csv data_BLR/map/segments.csv

	It requires installation of Julia 0.3 with OpenStreetMap.jl extension - https://github.com/tedsteiner/OpenStreetMap.jl
	A new version package is available for Julia 1.0 at https://github.com/pszufe/OpenStreetMapX.jl it may be used.

## Modifying nodes and segments file
	The format of above generated files is modified using simple find/replace as :

		nodes.csv :
		1522655984,LLA(12.9674517,77.5624332,0.0)
		1588453609,LLA(12.8820959,77.4928339,0.0)
		2450408122,LLA(13.0411969,77.5809113,0.0)
		.
		.
		
		nodes_mod.csv :
		id,lat,lon,al
		1522655984,12.9674517,77.5624332,0.0
		1588453609,12.8820959,77.4928339,0.0
		2450408122,13.0411969,77.5809113,0.0
		.
		.
		
		segments.csv :
		Segment(419638272,599212733,[419638272,599212733],70.86635228852029,6,46951459,true)
		Segment(599212733,419638272,[599212733,419638272],70.86635228852029,6,46951459,true)
		Segment(599212733,599212731,[599212733,599212731],53.45547791675319,6,46951459,true)
		.
		.
		
		segments_mod.csv :
		u,v,list,dist,int,pid,oneway
		419638272,599212733,"[419638272,599212733]",70.86635228852029,6,46951459,true
		599212733,419638272,"[599212733,419638272]",70.86635228852029,6,46951459,true
		599212733,599212731,"[599212733,599212731]",53.45547791675319,6,46951459,true
		.
		.
		
		Files Generated : data_BLR/map/nodes_mod.csv data_BLR/map/segments_mod.csv

## Modifying Map
	The julia script extracts roads compatible for vehicles and marks intersection of these roads as nodes of a map.
	We need to modify our map according to these nodes and edges for further applications.
	Use scripts/modify_map.py for the purpose.

	python3 scripts/modify_map.py
	
	Files Generated : data_BLR/map/map_mod.xml
	
## Generating Edge Speeds using GraphHopper
	This is one-time time consuming process - Use the processed file already provided
	
	We use GraphHopper : https://github.com/graphhopper/map-matching for map matching
	Install the package and import the modified map to create a new "graph-cache"
	Use scripts/get_edge_weights.py for edge speed generation

	python3 scripts/get_edge_weights.py

	Files Generated : data_BLR/map/edge_weights.pkl

## Map Connectivity
	The OSM map obtained may not be strongly connected. 
	scripts/get_connected_map.py extracts the largest strongly connected component of the graph
	Use the connected version if required.
	
	 python3 scripts/get_connected_map.py

	Files Generated : data_BLR/map/nodes_mod_connected.csv data_BLR/map/segments_mod_connected.csv

## Orders and Ping data
	Lat/long values need to be mapped to corresponding closest nodes. 
	scripts/order_tables_nodeid.py and scripts/rest_de_ping_node_id.py do the same. 
	
	 python3 scripts/order_tables_nodeid.py
	python3 scripts/rest_de_ping_node_id.py

## Generating Data files 
	Use scripts/create_cpp_data_files.py to generate all the files required by the CPP code
	
	python3 scripts/create_cpp_data_files.py

## HHL Graph Index
	git clone https://github.com/savrus/hl.git
	make

	Generate order file
	This is one-time time consuming process
    ./hhl -o ~/data_BLR/map/cpp_code/dimacs_0.order ~/data_BLR/map/cpp_code/per_hour_edges/dimacs_0

	Generate HHL labels
	bash scripts/gen_hhl_label.sh
	
# Data Preparation Complete

