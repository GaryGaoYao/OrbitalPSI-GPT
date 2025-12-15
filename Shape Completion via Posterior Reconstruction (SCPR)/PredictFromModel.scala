//> using scala "3.3"
//> using dep "ch.unibas.cs.gravis::scalismo-ui:0.92.0"

import scalismo.ui.api.*
import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.io.{MeshIO, StatisticalModelIO}

import java.io.File
import scala.util.{Try, Success, Failure}

object PredictFromModel extends App:

  scalismo.initialize()
  val ui = ScalismoUI()

  // -----------------------
  // Args (simple)
  // -----------------------
  def getArg(key: String, default: String): String =
    val idx = args.indexOf(key)
    if (idx >= 0 && idx + 1 < args.length) args(idx + 1) else default

  val modelPath  = getArg("--model",  "output/gpmm_region_aware/region_aware_gpmm.h5")
  val targetPath = getArg("--target", "Orbital-Bone-Dataset/subject_01.stl")
  val outDirPath = getArg("--out",    "output/predict")

  val outDir = new File(outDirPath)
  if (!outDir.exists()) outDir.mkdirs()

  val outPred = new File(outDir, "predicted.stl")

  // -----------------------
  // Load model + target
  // -----------------------
  val model = StatisticalModelIO.readStatisticalTriangleMeshModel3D(new File(modelPath)) match
    case Success(m) => m
    case Failure(e) => throw new RuntimeException(s"Failed to read model: ${e.getMessage}")

  val target = MeshIO.readMesh(new File(targetPath)) match
    case Success(m) => m
    case Failure(e) => throw new RuntimeException(s"Failed to read target: ${e.getMessage}")

  // topology check
  require(
    target.triangulation == model.reference.triangulation,
    "Target mesh topology != model reference topology. " +
    "Prediction requires correspondence (same triangulation)."
  )

  // -----------------------
  // Reflection-based predict
  // -----------------------
  def tryInvoke1(obj: AnyRef, name: String, arg: AnyRef): Option[AnyRef] =
    val methods = obj.getClass.getMethods.filter(m => m.getName == name && m.getParameterCount == 1)
    methods.headOption.flatMap { m =>
      Try(m.invoke(obj, arg)).toOption.map(_.asInstanceOf[AnyRef])
    }

  // 1) try project(target)
  val predictedOpt1 =
    tryInvoke1(model.asInstanceOf[AnyRef], "project", target.asInstanceOf[AnyRef])
      .map(_.asInstanceOf[TriangleMesh[_3D]])

  // 2) try coefficients(target) + instance(coeffs)
  val predictedOpt2 =
    if (predictedOpt1.isDefined) None
    else
      val coeffsOpt = tryInvoke1(model.asInstanceOf[AnyRef], "coefficients", target.asInstanceOf[AnyRef])
      coeffsOpt.flatMap { coeffs =>
        tryInvoke1(model.asInstanceOf[AnyRef], "instance", coeffs)
          .map(_.asInstanceOf[TriangleMesh[_3D]])
      }

  val predicted =
    predictedOpt1.orElse(predictedOpt2).getOrElse {
      throw new RuntimeException(
        "Could not predict via model.project(target) nor model.coefficients/instance. " +
        "Please tell me the exact Scalismo API error/output, I will adapt this script to your local build."
      )
    }

  // save
  MeshIO.writeMesh(predicted, outPred) match
    case Success(_) => println(s"✅ Predicted STL saved: ${outPred.getPath}")
    case Failure(e) => throw new RuntimeException(s"Failed to write predicted STL: ${e.getMessage}")

  // visualize
  val g = ui.createGroup("Prediction")
  ui.show(g, model.mean, "Mean")
  ui.show(g, target, "Target")
  ui.show(g, predicted, "Predicted")

  println("✅ Done.")
