package xgboost

/*
#include <stdlib.h>

#include <xgboost/c_api.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Booster is a Go representation of the XGBooster C type.
type Booster struct {
	handle C.BoosterHandle
}

// NewBooster creates a new Booster.
func NewBooster(dmats []*DMatrix) (*Booster, error) {
	bst := new(Booster)

	if len(dmats) == 0 {
		// Handle the case where dmats is empty. You could return an error
		// here, or you could allow an empty slice and skip the XGBoosterCreate
		// call if dmats is empty.
		return nil, fmt.Errorf("dmats must not be empty")
	}

	cDmats := make([]C.DMatrixHandle, len(dmats))
	for i, dmat := range dmats {
		cDmats[i] = dmat.handle
	}

	ret := C.XGBoosterCreate(&cDmats[0], C.bst_ulong(len(dmats)), &bst.handle)

	if ret != 0 {
		return nil, fmt.Errorf("could not create booster: %w", xgbError(ret))
	}

	return bst, nil
}

// Free frees the memory of the Booster.
func (bst *Booster) Free() error {
	return xgbError(C.XGBoosterFree(bst.handle))
}

// Slice creates a new Booster that includes only a specific subset of the trees from the original Booster.
func (bst *Booster) Slice(beginLayer int, endLayer int, step int) (*Booster, error) {
	newBst := new(Booster)
	ret := C.XGBoosterSlice(bst.handle, C.int(beginLayer), C.int(endLayer), C.int(step), &newBst.handle)
	if err := xgbError(ret); err != nil {
		return nil, err
	}
	return newBst, nil
}

// BoostedRounds returns the number of boosting rounds that have been performed.
func (bst *Booster) BoostedRounds() (int, error) {
	var out C.int
	ret := C.XGBoosterBoostedRounds(bst.handle, &out)
	if err := xgbError(ret); err != nil {
		return 0, err
	}
	return int(out), nil
}

// SetParam sets a parameter of the Booster.
func (bst *Booster) SetParam(name string, value string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	cValue := C.CString(value)
	defer C.free(unsafe.Pointer(cValue))

	return xgbError(C.XGBoosterSetParam(bst.handle, cName, cValue))
}

// GetNumFeature retrieves the number of features.
func (bst *Booster) GetNumFeature() (num_feature uint, err error) {
	var out C.bst_ulong
	ret := C.XGBoosterGetNumFeature(bst.handle, &out)
	if err = xgbError(ret); err != nil {
		return 0, err
	}
	return uint(out), nil
}

// UpdateOneIter performs one iteration of boosting.
func (bst *Booster) UpdateOneIter(iter int, dtrain *DMatrix) error {
	return xgbError(C.XGBoosterUpdateOneIter(bst.handle, C.int(iter), dtrain.handle))
}

// BoostOneIter boosts one iteration with custom gradient statistics.
// The gradient statistics should be packed into pairs of (grad, hess).
func (bst *Booster) BoostOneIter(dtrain *DMatrix, grad []float32, hess []float32) error {
	// Check the lengths of grad and hess.
	if len(grad) != len(hess) {
		return fmt.Errorf("grad and hess must have the same length")
	}

	return xgbError(C.XGBoosterBoostOneIter(bst.handle, dtrain.handle, (*C.float)(&grad[0]), (*C.float)(&hess[0]), C.bst_ulong(len(grad))))
}

// EvalOneIter evaluates the performance of the model after a given number of iterations.
func (bst *Booster) EvalOneIter(iter int, dmats []*DMatrix, evnames []string) (string, error) {
	cDmats := make([]C.DMatrixHandle, len(dmats))
	for i, dmat := range dmats {
		cDmats[i] = dmat.handle
	}

	cEvnames := make([]*C.char, len(evnames))
	for i, name := range evnames {
		cEvnames[i] = C.CString(name)
		defer C.free(unsafe.Pointer(cEvnames[i]))
	}

	var outResult *C.char
	ret := C.XGBoosterEvalOneIter(bst.handle, C.int(iter), &cDmats[0], &cEvnames[0], C.bst_ulong(len(dmats)), &outResult)
	if err := xgbError(ret); err != nil {
		return "", err
	}

	return C.GoString(outResult), nil
}

// Predict performs prediction.
func (bst *Booster) Predict(dmat *DMatrix, optionMask int, ntreeLimit uint, training bool) ([]float32, error) {
	var outLen C.bst_ulong
	var outResult *C.float
	ret := C.XGBoosterPredict(bst.handle, dmat.handle, C.int(optionMask), C.uint(ntreeLimit),
		C.int(boolToInt(training)), &outLen, &outResult)

	if err := xgbError(ret); err != nil {
		return nil, err
	}

	resultSlice := (*[1 << 30]float32)(unsafe.Pointer(outResult))[:outLen:outLen]
	return resultSlice, nil
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func (bst *Booster) PredictFromDMatrix(dmat *DMatrix, config string) ([]float32, error) {
	cConfig := C.CString(config)
	defer C.free(unsafe.Pointer(cConfig))

	var outDim C.bst_ulong
	var outShape *C.bst_ulong
	var outResult *C.float
	ret := C.XGBoosterPredictFromDMatrix(bst.handle, dmat.handle, cConfig, &outShape, &outDim, &outResult)

	if err := xgbError(ret); err != nil {
		return nil, err
	}

	var resultSlice []float32
	return resultSlice, fmt.Errorf("not implemented")
}

// PredictFromDense predicts the labels from a given dense array.
func (bst *Booster) PredictFromDense(values string, config string, m *DMatrix) ([]float32, error) {
	cValues := C.CString(values)
	defer C.free(unsafe.Pointer(cValues))

	cConfig := C.CString(config)
	defer C.free(unsafe.Pointer(cConfig))

	var outShape *C.bst_ulong
	var outDim C.bst_ulong
	var outResult *C.float
	ret := C.XGBoosterPredictFromDense(bst.handle, cValues, cConfig, m.handle, &outShape, &outDim, &outResult)

	if err := xgbError(ret); err != nil {
		return nil, err
	}

	var resultSlice []float32
	return resultSlice, fmt.Errorf("not implemented")
}

// PredictFromCSR predicts the labels from a given CSR format.
func (bst *Booster) PredictFromCSR(indptr string, indices string, values string, ncol uint, config string, m *DMatrix) ([]float32, error) {
	cIndptr := C.CString(indptr)
	defer C.free(unsafe.Pointer(cIndptr))

	cIndices := C.CString(indices)
	defer C.free(unsafe.Pointer(cIndices))

	cValues := C.CString(values)
	defer C.free(unsafe.Pointer(cValues))

	cConfig := C.CString(config)
	defer C.free(unsafe.Pointer(cConfig))

	var outShape *C.bst_ulong
	var outDim C.bst_ulong
	var outResult *C.float
	ret := C.XGBoosterPredictFromCSR(bst.handle, cIndptr, cIndices, cValues, C.bst_ulong(ncol), cConfig, m.handle, &outShape, &outDim, &outResult)

	if err := xgbError(ret); err != nil {
		return nil, err
	}

	var resultSlice []float32
	return resultSlice, fmt.Errorf("not implemented")
}

// TODO XGBoosterPredictFromCudaArray
// TODO XGBoosterPredictFromCudaColumnar

// ======================================

// SaveModelToBuffer saves the model to a buffer.
func (bst *Booster) SaveModelToBuffer(config string) ([]byte, error) {
	var outLen C.bst_ulong
	var outResult *C.char
	cConfig := C.CString(config)
	defer C.free(unsafe.Pointer(cConfig))
	ret := C.XGBoosterSaveModelToBuffer(bst.handle, cConfig, &outLen, &outResult)
	if err := xgbError(ret); err != nil {
		return nil, err
	}
	resultSlice := C.GoBytes(unsafe.Pointer(outResult), C.int(outLen))
	return resultSlice, nil
}

// GetModelRaw returns the raw model.
func (bst *Booster) GetModelRaw() ([]byte, error) {
	var outLen C.bst_ulong
	var outResult *C.char
	ret := C.XGBoosterGetModelRaw(bst.handle, &outLen, &outResult)
	if err := xgbError(ret); err != nil {
		return nil, err
	}
	resultSlice := C.GoBytes(unsafe.Pointer(outResult), C.int(outLen))
	return resultSlice, nil
}

// LoadJsonConfig loads a JSON config.
func (bst *Booster) LoadJsonConfig(config string) error {
	cConfig := C.CString(config)
	defer C.free(unsafe.Pointer(cConfig))
	return xgbError(C.XGBoosterLoadJsonConfig(bst.handle, cConfig))
}

// SetStrFeatureInfo sets a string feature.
func (bst *Booster) SetStrFeatureInfo(field string, features []string) error {
	cField := C.CString(field)
	defer C.free(unsafe.Pointer(cField))
	cFeatures := make([]*C.char, len(features))
	for i := range features {
		cFeatures[i] = C.CString(features[i])
		defer C.free(unsafe.Pointer(cFeatures[i]))
	}
	return xgbError(C.XGBoosterSetStrFeatureInfo(bst.handle, cField, &cFeatures[0], C.bst_ulong(len(features))))
}

// GetStrFeatureInfo gets a string feature.
func (bst *Booster) GetStrFeatureInfo(field string) ([]string, error) {
	cField := C.CString(field)
	defer C.free(unsafe.Pointer(cField))
	var outLen C.bst_ulong
	var outResult **C.char
	ret := C.XGBoosterGetStrFeatureInfo(bst.handle, cField, &outLen, &outResult)
	if err := xgbError(ret); err != nil {
		return nil, err
	}
	resultSlice := make([]string, outLen)
	cArray := (*[1 << 30]*C.char)(unsafe.Pointer(outResult))
	for i := 0; i < int(outLen); i++ {
		resultSlice[i] = C.GoString(cArray[i])
	}
	return resultSlice, nil
}

// SerializeToBuffer serializes the booster to a buffer.
func (bst *Booster) SerializeToBuffer() ([]byte, error) {
	var outLen C.bst_ulong
	var outDptr *C.char
	ret := C.XGBoosterSerializeToBuffer(bst.handle, &outLen, &outDptr)
	if err := xgbError(ret); err != nil {
		return nil, err
	}
	return C.GoBytes(unsafe.Pointer(outDptr), C.int(outLen)), nil
}

// UnserializeFromBuffer unserializes the booster from a buffer.
func (bst *Booster) UnserializeFromBuffer(buf []byte) error {
	return xgbError(C.XGBoosterUnserializeFromBuffer(bst.handle, unsafe.Pointer(&buf[0]), C.bst_ulong(len(buf))))
}

// SaveJsonConfig saves the configuration to a JSON string.
func (bst *Booster) SaveJsonConfig() (string, error) {
	var outLen C.bst_ulong
	var outStr *C.char
	ret := C.XGBoosterSaveJsonConfig(bst.handle, &outLen, &outStr)
	if err := xgbError(ret); err != nil {
		return "", err
	}
	return C.GoString(outStr), nil
}

// DumpModel dumps the model.
func (bst *Booster) DumpModel(fmap string, withStats bool) ([]string, error) {
	cFmap := C.CString(fmap)
	defer C.free(unsafe.Pointer(cFmap))
	var outLen C.bst_ulong
	var outDumpArray **C.char
	ret := C.XGBoosterDumpModel(bst.handle, cFmap, C.int(boolToInt(withStats)), &outLen, &outDumpArray)
	if err := xgbError(ret); err != nil {
		return nil, err
	}
	dumpArraySlice := (*[1 << 30]*C.char)(unsafe.Pointer(outDumpArray))[:outLen:outLen]
	result := make([]string, outLen)
	for i := 0; i < int(outLen); i++ {
		result[i] = C.GoString(dumpArraySlice[i])
	}
	return result, nil
}

// FeatureScore computes feature importance scores.
func (bst *Booster) FeatureScore(config string) ([]string, []uint, []float32, error) {
	cConfig := C.CString(config)
	defer C.free(unsafe.Pointer(cConfig))
	var outNFeatures C.bst_ulong
	var outFeatures **C.char
	var outDim C.bst_ulong
	var outShape *C.bst_ulong
	var outScores *C.float
	ret := C.XGBoosterFeatureScore(bst.handle, cConfig, &outNFeatures, &outFeatures, &outDim, &outShape, &outScores)
	if err := xgbError(ret); err != nil {
		return nil, nil, nil, err
	}
	featuresSlice := (*[1 << 30]*C.char)(unsafe.Pointer(outFeatures))[:outNFeatures:outNFeatures]
	features := make([]string, outNFeatures)
	for i := 0; i < int(outNFeatures); i++ {
		features[i] = C.GoString(featuresSlice[i])
	}
	shapeSlice := (*[1 << 30]uint)(unsafe.Pointer(outShape))[:outDim:outDim]
	scoresSlice := (*[1 << 30]float32)(unsafe.Pointer(outScores))[:outDim:outDim]
	return features, shapeSlice, scoresSlice, nil
}

// DumpModelEx dumps the model with additional options.
func (bst *Booster) DumpModelEx(fmap string, withStats bool, format string) ([]string, error) {
	cFmap := C.CString(fmap)
	defer C.free(unsafe.Pointer(cFmap))
	cFormat := C.CString(format)
	defer C.free(unsafe.Pointer(cFormat))
	var outLen C.bst_ulong
	var outDumpArray **C.char
	ret := C.XGBoosterDumpModelEx(bst.handle, cFmap, C.int(boolToInt(withStats)), cFormat, &outLen, &outDumpArray)
	if err := xgbError(ret); err != nil {
		return nil, err
	}
	dumpArraySlice := (*[1 << 30]*C.char)(unsafe.Pointer(outDumpArray))[:outLen:outLen]
	result := make([]string, outLen)
	for i := 0; i < int(outLen); i++ {
		result[i] = C.GoString(dumpArraySlice[i])
	}
	return result, nil
}

// DumpModelWithFeatures dumps the model along with feature names.
func (bst *Booster) DumpModelWithFeatures(fnum int, fname []string, ftype []string, withStats bool) ([]string, error) {
	cFname := make([]*C.char, len(fname))
	for i, v := range fname {
		cFname[i] = C.CString(v)
		defer C.free(unsafe.Pointer(cFname[i]))
	}
	cFtype := make([]*C.char, len(ftype))
	for i, v := range ftype {
		cFtype[i] = C.CString(v)
		defer C.free(unsafe.Pointer(cFtype[i]))
	}
	var outLen C.bst_ulong
	var outDumpArray **C.char
	ret := C.XGBoosterDumpModelWithFeatures(bst.handle, C.int(fnum), &cFname[0], &cFtype[0], C.int(boolToInt(withStats)), &outLen, &outDumpArray)
	if err := xgbError(ret); err != nil {
		return nil, err
	}
	dumpArraySlice := (*[1 << 30]*C.char)(unsafe.Pointer(outDumpArray))[:outLen:outLen]
	result := make([]string, outLen)
	for i := 0; i < int(outLen); i++ {
		result[i] = C.GoString(dumpArraySlice[i])
	}
	return result, nil
}

// DumpModelExWithFeatures dumps the model along with feature names and additional options.
func (bst *Booster) DumpModelExWithFeatures(fnum int, fname []string, ftype []string, withStats bool, format string) ([]string, error) {
	cFname := make([]*C.char, len(fname))
	for i, v := range fname {
		cFname[i] = C.CString(v)
		defer C.free(unsafe.Pointer(cFname[i]))
	}
	cFtype := make([]*C.char, len(ftype))
	for i, v := range ftype {
		cFtype[i] = C.CString(v)
		defer C.free(unsafe.Pointer(cFtype[i]))
	}
	cFormat := C.CString(format)
	defer C.free(unsafe.Pointer(cFormat))
	var outLen C.bst_ulong
	var outDumpArray **C.char
	ret := C.XGBoosterDumpModelExWithFeatures(bst.handle, C.int(fnum), &cFname[0], &cFtype[0], C.int(boolToInt(withStats)), cFormat, &outLen, &outDumpArray)
	if err := xgbError(ret); err != nil {
		return nil, err
	}
	dumpArraySlice := (*[1 << 30]*C.char)(unsafe.Pointer(outDumpArray))[:outLen:outLen]
	result := make([]string, outLen)
	for i := 0; i < int(outLen); i++ {
		result[i] = C.GoString(dumpArraySlice[i])
	}
	return result, nil
}

// LoadModelFromBuffer loads the model from a buffer.
func (bst *Booster) LoadModelFromBuffer(buf []byte) error {
	return xgbError(C.XGBoosterLoadModelFromBuffer(bst.handle, unsafe.Pointer(&buf[0]), C.bst_ulong(len(buf))))
}

// GetAttr retrieves an attribute.
func (bst *Booster) GetAttr(key string) (value string, exists bool, err error) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var out *C.char
	var success *C.int
	ret := C.XGBoosterGetAttr(bst.handle, cKey, &out, success)
	if err = xgbError(ret); err != nil {
		return "", false, err
	}
	return C.GoString(out), true, nil
}

// SetAttr sets an attribute.
func (bst *Booster) SetAttr(key string, value string) error {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	cValue := C.CString(value)
	defer C.free(unsafe.Pointer(cValue))

	return xgbError(C.XGBoosterSetAttr(bst.handle, cKey, cValue))
}

// GetAttrNames retrieves all attribute names.
func (bst *Booster) GetAttrNames() (names []string, err error) {
	var num_out C.bst_ulong
	var s **C.char
	ret := C.XGBoosterGetAttrNames(bst.handle, &num_out, &s)
	if err = xgbError(ret); err != nil {
		return nil, err
	}

	names = make([]string, num_out)
	cArray := (*[1 << 30]*C.char)(unsafe.Pointer(s))

	for i := 0; i < int(num_out); i++ {
		names[i] = C.GoString(cArray[i])
	}

	return names, nil
}

// LoadModel loads a model from a file.
func (bst *Booster) LoadModel(fname string) error {
	cFname := C.CString(fname)
	defer C.free(unsafe.Pointer(cFname))

	return xgbError(C.XGBoosterLoadModel(bst.handle, cFname))
}

// SaveModel saves the model to a file.
func (bst *Booster) SaveModel(fname string) error {
	cFname := C.CString(fname)
	defer C.free(unsafe.Pointer(cFname))

	return xgbError(C.XGBoosterSaveModel(bst.handle, cFname))
}
