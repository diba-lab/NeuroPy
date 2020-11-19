if [ ! -f ./spykcirc ]
	then
		mkdir ./spykcirc
fi

for name in cluster_info.tsv spike_clusters.npy spike_times.npy
	do
		find ./ -name $name | xargs ln -s
	done

