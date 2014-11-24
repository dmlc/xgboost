# a simple script to simulate mapreduce mapper
echo "cat $1$OMPI_COMM_WORLD_RANK | ${@:2}"
cat $1$OMPI_COMM_WORLD_RANK | ${@:2}
