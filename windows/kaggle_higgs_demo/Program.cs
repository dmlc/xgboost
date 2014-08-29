using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using xgboost_sharp_wrapper;

namespace kaggle_higgs_demo
{
    class Program
    {
        static void Main(string[] args)
        {
            string usage_str = "training_path.libsvm test_path.libsvm sharp_pred.csv";
            string[] usage_strings = usage_str.Split(' ');
            if (args.Length != usage_strings.Length)
            {
                Console.WriteLine("Usage: " + usage_str);
                Console.WriteLine("wrong number of arguments");
                return;
            }
            Dictionary<string, string> argsDictionary = new Dictionary<string, string>();
            for (int i = 0; i < args.Length; i++)
            {
                argsDictionary.Add(usage_strings[i], args[i]);
            }
            try
            {
                kaggle_higgs_demo(argsDictionary);
            }
            catch (Exception exc)
            {
                Console.WriteLine(exc.Message);
            }
        }

        private static void kaggle_higgs_demo(Dictionary<string, string> argsDictionary)
        {
            xgboost xgb = new xgboost();
            IntPtr dtrain = xgb.SharpXGDMatrixCreateFromFile( (argsDictionary["training_path.libsvm"]), 0);
            IntPtr[] dmats = new IntPtr[1];
            dmats[0] = dtrain;
            Console.WriteLine("loading data end, start to boost trees");
            xgb.SharpXGBoosterCreate(dmats,1);
            Console.WriteLine("booster created");
        }

        static byte[] GetBytes(string str)
        {
            byte[] bytes = new byte[str.Length * sizeof(char)];
            System.Buffer.BlockCopy(str.ToCharArray(), 0, bytes, 0, bytes.Length);
            return bytes;
        }

        static string GetString(byte[] bytes)
        {
            char[] chars = new char[bytes.Length / sizeof(char)];
            System.Buffer.BlockCopy(bytes, 0, chars, 0, bytes.Length);
            return new string(chars);
        }
    }
}
