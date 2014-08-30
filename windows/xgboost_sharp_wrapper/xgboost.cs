using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace xgboost_sharp_wrapper
{
    
        
    public class xgboost
    {
        private const string dll_path="..\\x64\\Release\\";

        [DllImport(dll_path+"xgboost_wrapper.dll", CallingConvention=CallingConvention.Cdecl)]
        public static extern IntPtr XGDMatrixCreateFromFile(string fname, int silent);
public IntPtr SharpXGDMatrixCreateFromFile(string fname, int silent)
        {
            return XGDMatrixCreateFromFile(fname, silent);
        }

        [DllImport(dll_path + "xgboost_wrapper.dll")]
        public static extern IntPtr XGBoosterCreate(IntPtr[] dmats, System.UInt32 len);
        public IntPtr SharpXGBoosterCreate(IntPtr[] dmats, System.UInt32 len)
        {
            return XGBoosterCreate(dmats, len);
        }

        [DllImport(dll_path + "xgboost_wrapper.dll")]
        public static extern void XGBoosterUpdateOneIter(IntPtr handle, int iter, IntPtr dtrain);
        public void SharpXGBoosterUpdateOneIter(IntPtr handle, int iter, IntPtr dtrain)
        {
            XGBoosterUpdateOneIter(handle, iter, dtrain);
        }

        [DllImport(dll_path + "xgboost_wrapper.dll")]
        public static extern string XGBoosterEvalOneIter(IntPtr handle, int iter, IntPtr[] dmats,
                                           string[] evnames, System.UInt32 len);
        public string SharpXGBoosterEvalOneIter(IntPtr handle, int iter, IntPtr[] dmats,
                                           string[] evnames, System.UInt32 len)
        {
            return XGBoosterEvalOneIter(handle, iter, dmats, evnames, len);
        }
  /*!
   * \brief make prediction based on dmat
   * \param handle handle
   * \param dmat data matrix
   * \param output_margin whether only output raw margin value
   * \param len used to store length of returning result
   */
        [DllImport(dll_path + "xgboost_wrapper.dll")]
        public static extern IntPtr XGBoosterPredict(IntPtr handle, IntPtr dmat, int output_margin, ref System.UInt32 len);
        
        public float[] SharpXGBoosterPredict(IntPtr handle, IntPtr dmat, int output_margin, System.UInt32 len)
        {
            IntPtr buf = XGBoosterPredict(handle, dmat, output_margin, ref len);
            
            float[] buffer = new float[len];

            Marshal.Copy(buf, buffer, 0, buffer.Length);

            return buffer;
        }
  /*!
   * \brief load model from existing file
   * \param handle handle
   * \param fname file name
   */
        [DllImport(dll_path + "xgboost_wrapper.dll")]
        public static extern void XGBoosterLoadModel(IntPtr handle, string fname);
  /*!
   * \brief save model into existing file
   * \param handle handle
   * \param fname file name
   */
        [DllImport(dll_path + "xgboost_wrapper.dll")]
        public static extern void XGBoosterSaveModel(IntPtr handle, string fname);
  /*!
   * \brief dump model, return array of strings representing model dump
   * \param handle handle
   * \param fmap  name to fmap can be empty string
   * \param out_len length of output array
   * \return char *data[], representing dump of each model
   */
        [DllImport(dll_path + "xgboost_wrapper.dll")]
        public static extern string XGBoosterDumpModel(IntPtr handle, string fmap,
                                           System.UInt32 out_len);

    }
}
