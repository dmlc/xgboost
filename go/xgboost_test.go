package xgboost_test

import (
	"testing"

	xgboost "github.com/dmlc/xgboost/go"
)

func TestXGBoostVersion(t *testing.T) {
	t.Parallel()
	v1, v2, v3 := xgboost.XGBoostVersion()
	t.Logf("XGBoostVersion() = %v.%v.%v", v1, v2, v3)
}

func TestXGBuildInfo(t *testing.T) {
	t.Parallel()
	buildInfo := xgboost.XGBuildInfo()
	t.Logf("XGBuildInfo() = %v", buildInfo)
}

func TestNewBooster(t *testing.T) {
	// This is the table of test cases.
	testCases := []struct {
		name    string
		dmats   []*xgboost.DMatrix
		wantErr bool
	}{
		{
			name:    "empty dmats slice",
			dmats:   []*xgboost.DMatrix{},
			wantErr: true,
		},
		// Add more test cases here...
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			booster, err := xgboost.NewBooster(tc.dmats)

			// Check if an error was returned.
			if (err != nil) != tc.wantErr {
				t.Errorf("NewBooster() error = %v, wantErr %v", err, tc.wantErr)
			}

			// Check if a nil Booster was returned.
			if err == nil && booster == nil {
				t.Error("NewBooster() returned a nil Booster and nil error")
			}
		})
	}
}
