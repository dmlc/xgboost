
PLUGIN_OBJS += build_plugin/updater_gpu/src/register_updater_gpu.o \
               build_plugin/updater_gpu/src/updater_gpu.o \
               build_plugin/updater_gpu/src/gpu_hist_builder.o \
               build_plugin/updater_gpu/src/gpu_predictor.o
PLUGIN_LDFLAGS += -L$(CUDA_ROOT)/lib64 -lcudart
