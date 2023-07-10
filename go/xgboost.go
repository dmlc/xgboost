package xgboost

/*
#cgo CFLAGS: -I../include
#cgo LDFLAGS: -L../lib -lxgboost -Wl,-rpath,../lib
#include <stdlib.h>
#include <xgboost/c_api.h>

extern void bridge_log_callback(const char *msg);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// XGBoostVersion retrieves the version of the underlying XGBoost library.
func XGBoostVersion() (int, int, int) {
	var major C.int
	var minor C.int
	var patch C.int
	C.XGBoostVersion(&major, &minor, &patch)
	return int(major), int(minor), int(patch)
}

// XGBSetGlobalConfig sets the global configuration.
func XGBSetGlobalConfig(config string) C.int {
	cConfig := C.CString(config)
	defer C.free(unsafe.Pointer(cConfig))
	return C.XGBSetGlobalConfig(cConfig)
}

// XGBGetGlobalConfig retrieves the global configuration.
func XGBGetGlobalConfig() string {
	var outConfig *C.char
	C.XGBGetGlobalConfig(&outConfig)
	return C.GoString(outConfig)
}

// XGBuildInfo retrieves the build information of the underlying XGBoost library.
func XGBuildInfo() string {
	var out *C.char
	C.XGBuildInfo(&out)
	return C.GoString(out)
}

var goLogCallbackGoFunc func(msg string)

//export goLogCallbackFunc
func goLogCallbackFunc(msg *C.char) {
	if goLogCallbackGoFunc != nil {
		goLogCallbackGoFunc(C.GoString(msg))
	}
}

// XGBRegisterLogCallback registers a log callback function.
func XGBRegisterLogCallback(callback func(msg string)) {
	goLogCallbackGoFunc = callback
	C.XGBRegisterLogCallback((*[0]byte)(C.bridge_log_callback))
}

func xgbError(ret C.int) error {
	if ret == 0 {
		return nil
	}
	// get last error:
	le := C.XGBGetLastError()

	// Convert the error code to a string and return it as an error.
	// This is a very basic error handling strategy.
	return fmt.Errorf("xgboost: %s", C.GoString(le))
}
