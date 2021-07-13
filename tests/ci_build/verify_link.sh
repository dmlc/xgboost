# Make sure the dependencies of XGBoost don't appear in directly downstream project.
# Pass the executable as argument for this script

if readelf -d $1 | grep "omp";
then
    echo "Found openmp in direct dependency"
    exit -1
else
    exit 0
fi

if readelf -d $1 | grep "pthread";
then
    echo "Found pthread in direct dependency"
    exit -1
else
    exit 0
fi
