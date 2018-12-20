using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Step01ConsoleApp
{
    public static class IrisTextLoaderFactory
    {
        public static TextLoader CreateTextLoader(MLContext mlContext)
        {
            TextLoader textLoader = mlContext.Data.TextReader(
                new TextLoader.Arguments()
                {
                    SeparatorChars=new char[] { ',' },
                    HasHeader=false,
                    Column = new []
                    {
                        new TextLoader.Column("SepalLength", DataKind.R4, 0),
                        new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                        new TextLoader.Column("PetalLength", DataKind.R4, 2),
                        new TextLoader.Column("PetalWidth", DataKind.R4, 3),
                        new TextLoader.Column("Label", DataKind.TX, 4),
                    }
                });
            return textLoader;
        }
    }
}
