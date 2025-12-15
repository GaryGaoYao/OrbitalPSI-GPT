//> using scala "3.3"
//> using dep "ch.unibas.cs.gravis::scalismo-ui:0.92.0"

import scalismo.geometry.*
import scalismo.common.*
import scalismo.mesh.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.statisticalmodel.*
import scalismo.ui.api.*

import java.io.File
import scala.util.{Try, Success, Failure}

object PredictFromSSM extends App:

  // ---------------------------
  // 0) Init
  // ---------------------------
  scalismo.initialize()
  implicit val rng: scalismo.utils.Random = scalismo.utils.Random(42)
  val ui = ScalismoUI()

  // ---------------------------
  // ---------------------------
  val modelPath   = raw"D:\Codes\Scalismo\output\mySSM_model.h5"
  val targetStl   = raw"D:\Codes\Scalismo\some_new_registered_target.stl"

  val outDir = raw"D:\Codes\Scalismo\output\predict_exports"
  val outFolder = new File(outDir)
  if (!outFolder.exists()) outFolder.mkdirs()

  val predictedStlPath = new File(outFolder, "predicted.stl")
  val residualStlPath  = new File(outFolder, "residual_to_target.stl")

  // ---------------------------
  // 2) Load model
  // ---------------------------
  println("=== Loading SSM model ===")
  val ssm: StatisticalMeshModel =
    StatisticalModelIO.readStatisticalMeshModel(new File(modelPath)) match
      case Failure(ex) =>
        println(s"❌ Failed to read model: ${ex.getMessage}")
        sys.exit(1)
      case Success(m) =>
        println(s"✅ Model loaded. rank=${m.rank}")
        m

  // ---------------------------
  // 3) Load target mesh
  // ---------------------------
  println("=== Loading target mesh ===")
  val targetMesh = MeshIO.readMesh(new File(targetStl)).get
  println(s"✅ Target loaded: $targetStl")

  // 必须同拓扑
  require(
    targetMesh.triangulation == ssm.reference.triangulation,
    "❌ Target mesh topology != model reference topology！"
  )

  // ---------------------------
  // 4) Predict / reconstruct (estimate coefficients)
  // ---------------------------
  println("=== Estimating coefficients ===")

  val coeffsTry: Try[DenseVector[Double]] =
    Try(ssm.coefficients(targetMesh))

  val coeffs =
    coeffsTry match
      case Success(c) =>
        println(s"✅ Coefficients estimated. dim=${c.length}")
        c
      case Failure(ex) =>
        println(s"❌ Failed to estimate coefficients (likely API mismatch). Error: ${ex.getMessage}")
        sys.exit(1)

  val predictedMesh: TriangleMesh3D = ssm.instance(coeffs)

  // ---------------------------
  // 5) Save predicted STL
  // ---------------------------
  MeshIO.writeMesh(predictedMesh, predictedStlPath) match
    case Success(_) => println(s"✅ Predicted STL saved: ${predictedStlPath.getAbsolutePath}")
    case Failure(e) => println(s"❌ Failed to save predicted STL: ${e.getMessage}")

  MeshIO.writeMesh(targetMesh, new File(outFolder, "target.stl"))
  MeshIO.writeMesh(ssm.mean,   new File(outFolder, "mean.stl"))

  // ---------------------------
  // 6) Visualize
  // ---------------------------
  val group = ui.createGroup("Prediction")
  ui.show(group, ssm.mean, "Mean")
  ui.show(group, targetMesh, "Target")
  ui.show(group, predictedMesh, "Predicted")

  println("✅ Done.")
