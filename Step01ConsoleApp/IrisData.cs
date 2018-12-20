using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace Step01ConsoleApp
{
    //花的樣本
    public class IrisData
    {
        //萼片長度
        [Column("0")]
        public float SepalLength;
        //花萼寬度
        [Column("1")]
        public float SepalWidth;
        //花瓣長度
        [Column("2")]
        public float PetalLength;
        //花瓣寬度
        [Column("3")]
        public float PetalWidth;
        //品種
        [Column("4")]
        [ColumnName("Label")]
        public string Label;

    }
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
