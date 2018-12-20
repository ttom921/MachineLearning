using Microsoft.ML;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Data;
using System;
using Common;
using Microsoft.ML.Data;

namespace Step01ConsoleApp
{
    class Program
    {
        private static string TrainDataPath = "iris-train.txt";
        private static string ModelPath = "IrisClassificationModel.zip";
        static void Main(string[] args)
        {
            //創建學習管道對象
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 0);
            //1.
            BuildTrainEvaluateAndSaveModel(mlContext);
            //2.
            TestSomePredictions(mlContext);

            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
        private static void TestSomePredictions(MLContext mlContext)
        {
            //Test Classification Predictions with some hard-coded samples 

            var modelScorer = new Common.ModelScorer<IrisData, IrisPrediction>(mlContext);
            modelScorer.LoadModelFromZipFile(ModelPath);

        }
        private static void BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            // STEP 1: Common data loading configuration
            var textLoader = IrisTextLoaderFactory.CreateTextLoader(mlContext);
            var trainingDataView = textLoader.Read(TrainDataPath);
            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
            // Note that the label is text, so it needs to be converted to key.
            .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"), TransformerScope.TrainTest)
            // Use the multi-class SDCA model to predict the label using features.
           .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent())
            // Apply the inverse conversion from 'PredictedLabel' column back to string value.
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(("PredictedLabel", "PredictedLabel")));
            ;
            // STEP 3: Set the training algorithm, then create and config the modelBuilder                            
            var modelBuilder = new ModelBuilder<IrisData, IrisPrediction>(mlContext, dataProcessPipeline);
            // We apply our selected Trainer 
            var trainer = mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: "Label", featureColumn: "Features");
            modelBuilder.AddTrainer(trainer);
            // STEP 4: Train the model fitting to the DataSet
            //The pipeline is trained on the dataset that has been loaded and transformed.
            Console.WriteLine("=============== Training the model ===============");
            modelBuilder.Train(trainingDataView);


            // STEP 5: Evaluate the model and show accuracy stats
            //Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            //var metrics = modelBuilder.EvaluateMultiClassClassificationModel(testDataView, "Label");
            //Common.ConsoleHelper.PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);

            //// STEP 6: Save/persist the trained model to a .ZIP file
            Console.WriteLine("=============== Saving the model to a file ===============");
            modelBuilder.SaveModelAsFile(ModelPath);
        }
    }
}
