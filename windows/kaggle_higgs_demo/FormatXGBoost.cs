﻿using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace kaggle_higgs_demo
{
    static internal class FormatXGBoost
    {

        
        static internal double missing = -999.0;

        //---
        static internal void convert2xgboost(string training_path, string xgboost_training_path, string weight_training_path,
            out Dictionary<int, Event> EventsDictionary, bool is_cv, int test_from, int test_to, int test_length)
        {
            EventsDictionary = new Dictionary<int, Event>();
            string first_line;
            using (StreamReader sr = new StreamReader(training_path))
            {
                first_line = sr.ReadLine();
                int line = 0;
                string curr_line;
                while (!sr.EndOfStream)
                {
                    curr_line = sr.ReadLine();

                    Event evt = new Event();
                    string[] fields = curr_line.Split(',');
                    if (!int.TryParse(fields[0].Replace(".0",""), out evt.id))
                    {
                        Console.WriteLine("wrong id "+fields[0]+" for line " + line.ToString());
                    }
                    for (int ifeat = 1; ifeat <= fields.Length - 3; ifeat++)
                    {
                        double par_val;
                        if (double.TryParse(fields[ifeat], NumberStyles.Float, CultureInfo.InvariantCulture, out par_val))
                        {
                            evt.features.Add(par_val);

                        }
                        else
                        {
                            Console.WriteLine("Attention par_val not double: " + fields[ifeat]);
                            evt.features.Add(missing);
                            Console.WriteLine("Added missing !!!");
                        }
                    }
                    if (!double.TryParse(fields[fields.Length - 2], out evt.weight))
                    {
                        Console.WriteLine("wrong weight for line " + line.ToString());
                    }
                    evt.label = fields[fields.Length - 1];
                    EventsDictionary.Add(line++, evt);


                }
            }

            bool is_training = !(weight_training_path == null || weight_training_path.Trim().Length == 0);

            using (StreamWriter sw = new StreamWriter(xgboost_training_path, false))
            {
                for (int i = 0; i < EventsDictionary.Count; i++)
                {
                    if (is_cv)
                    {
                        if (is_training)
                        {
                            if (i >= test_from && i <= test_to)
                            {
                                continue;
                            }
                        }
                        else
                        {
                            if (i < test_from || i > test_to)
                            {
                                continue;
                            }
                        };
                    };

                    string curr_feature = "";
                    int ifeat = 0;
                    foreach (double par_val in EventsDictionary[i].features)
                    {
                        if (par_val != missing)
                        {
                            curr_feature += ifeat.ToString() + ":" + par_val.ToString(CultureInfo.InvariantCulture) + " ";
                        }

                        ifeat++;

                    }
                    sw.WriteLine(
                        (EventsDictionary[i].label.Equals("s")?"1":"0") + ' '
                        + curr_feature.TrimEnd()
                        );

                }
            }

            if (!is_training)
            {
                return;
            }

            using (StreamWriter sw = new StreamWriter(weight_training_path, false))
            {
                for (int i = 0; i < EventsDictionary.Count; i++)
                {
                    if (i >= test_from && i <= test_to)
                    {
                        continue;
                    }
                    sw.WriteLine(
                        (EventsDictionary[i].weight * (double)test_length / (double)(EventsDictionary.Count)).ToString(CultureInfo.InvariantCulture)
                        );

                }
            }


        }
        //---
    }
}
