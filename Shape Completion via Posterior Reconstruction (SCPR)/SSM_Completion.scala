//> using scala "3.3"
//> using dep "ch.unibas.cs.gravis::scalismo-ui:0.92.0"

import scalismo.geometry.*
import scalismo.common.*
import scalismo.mesh.*
import scalismo.statisticalmodel.MultivariateNormalDistribution
import scalismo.numerics.UniformMeshSampler3D
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.ui.api.*

import breeze.linalg.{DenseMatrix, DenseVector}

import java.io.File
import scala.io.Source
import scala.util.{Try, Success, Failure}

object Tutorial11_ExternalConstraints extends App:
  scalismo.initialize()
  implicit val rng: scalismo.utils.Random = scalismo.utils.Random(42)

  val ui = ScalismoUI()

  // -----------------------------
  // Defaults (can be overridden by CLI)
  // -----------------------------
  var targetPath = raw"D:\Codes\Scalismo\skull-model-oa-jianningli\ssm-test\A0001_clear.stl"
  var modelPath  = raw"output\SSM-orbital-bone.h5"

  var constraintsPath = raw"" // e.g., raw"D:\Codes\Scalismo\constraints.txt"

  var nSamplePts = 20000
  var nIterations = 50

  var icpSigma = 1.0         
  var constraintSigma = 0.2   

  var onlyConstraints = false

  // -----------------------------
  // Simple CLI parser
  // -----------------------------
  def popFlag(flag: String, args: Array[String]): Option[String] =
    val idx = args.indexOf(flag)
    if idx >= 0 && idx < args.length - 1 then Some(args(idx + 1)) else None

  def hasFlag(flag: String, args: Array[String]): Boolean =
    args.contains(flag)

  popFlag("--target", args).foreach(v => targetPath = v)
  popFlag("--model", args).foreach(v => modelPath = v)
  popFlag("--constraints", args).foreach(v => constraintsPath = v)
  popFlag("--samples", args).foreach(v => nSamplePts = v.toInt)
  popFlag("--iters", args).foreach(v => nIterations = v.toInt)
  popFlag("--icpSigma", args).foreach(v => icpSigma = v.toDouble)
  popFlag("--cSigma", args).foreach(v => constraintSigma = v.toDouble)
  if hasFlag("--onlyConstraints", args) then onlyConstraints = true

  // -----------------------------
  // Load target mesh & model
  // -----------------------------
  val targetMesh = MeshIO.readMesh(new File(targetPath)).get
  val model = StatisticalModelIO.readStatisticalTriangleMeshModel3D(new File(modelPath)).get

  val targetGroup = ui.createGroup("target")
  ui.show(targetGroup, targetMesh, "targetMesh")

  val modelGroup = ui.createGroup("model")
  ui.show(modelGroup, model, "model")

  // 必须拓扑一致
  require(
    targetMesh.triangulation == model.reference.triangulation,
    "❌ Topology mismatch: target mesh must have SAME triangulation as model.reference!"
  )

  // -----------------------------
  // Build sample point ids (for ICP correspondences)
  // -----------------------------
  val sampler = UniformMeshSampler3D(model.reference, numberOfPoints = nSamplePts)
  val sampledPoints: Seq[Point[_3D]] = sampler.samplePoints()
  val sampledPtIds: Seq[PointId] =
    sampledPoints.map(p => model.reference.pointSet.findClosestPoint(p).id)

  // -----------------------------
  // Read external constraints
  // -----------------------------
  def parseLineToNumbers(line: String): Array[Double] =
    line.trim
      .replace(",", " ")
      .split("\\s+")
      .filter(_.nonEmpty)
      .map(_.toDouble)

  /**
   * constraints.txt supports:
   *  - "x y z"       -> PointId from closest point on model.reference
   *  - "id x y z"    -> explicit PointId(id)
   */
  def readConstraints(path: String): IndexedSeq[(PointId, Point[_3D])] =
    if path == null || path.trim.isEmpty then IndexedSeq.empty
    else
      val src = Source.fromFile(path, "UTF-8")
      try
        src.getLines().toIndexedSeq
          .map(_.trim)
          .filter(l => l.nonEmpty && !l.startsWith("#"))
          .map { line =>
            val nums = parseLineToNumbers(line)
            if nums.length == 3 then
              val p = Point3D(nums(0), nums(1), nums(2))
              val id = model.reference.pointSet.findClosestPoint(p).id
              (id, p)
            else if nums.length >= 4 then
              val id = PointId(nums(0).toInt)
              val p = Point3D(nums(1), nums(2), nums(3))
              (id, p)
            else
              throw new IllegalArgumentException(s"Bad constraint line: '$line'")
          }
      finally src.close()

  val fixedConstraints: IndexedSeq[(PointId, Point[_3D])] = readConstraints(constraintsPath)
  println(s"✅ Loaded external constraints: ${fixedConstraints.size}")
  if fixedConstraints.nonEmpty then
    val cGroup = ui.createGroup("constraints")
    fixedConstraints.foreach { (id, p) =>
      val lm = Landmark(s"C_${id.id}", p)
      ui.show(cGroup, lm, s"constraint_${id.id}")
    }

  // -----------------------------
  // Helper: build correspondences for ICP
  // -----------------------------
  def attributeCorrespondences(movingMesh: TriangleMesh[_3D], ptIds: Seq[PointId]): IndexedSeq[(PointId, Point[_3D])] =
    ptIds.map { id =>
      val ptOnMoving = movingMesh.pointSet.point(id)
      val closestOnTarget = targetMesh.pointSet.findClosestPoint(ptOnMoving).point
      (id, closestOnTarget)
    }.toIndexedSeq

  // -----------------------------
  // Noise models
  // -----------------------------
  def isoNoise(sigma: Double): MultivariateNormalDistribution =
    val cov = DenseMatrix.eye * (sigma * sigma)
    MultivariateNormalDistribution(DenseVector.zeros, cov)

  val icpNoise = isoNoise(icpSigma)
  val constraintNoise = isoNoise(constraintSigma)

  // -----------------------------
  // Fit model with (ICP correspondences + external constraints)
  // -----------------------------
  def fitModel(
      icpCorr: IndexedSeq[(PointId, Point[_3D])],
      fixed: IndexedSeq[(PointId, Point[_3D])]
  ): TriangleMesh[_3D] =
    // 合并：若同一个 PointId 同时出现，外部 constraint 优先（更强、用户指定）
    val icpMap = icpCorr.toMap
    val fixedMap = fixed.toMap
    val mergedIds = (icpMap.keySet ++ fixedMap.keySet).toIndexedSeq

    val regressionData = mergedIds.map { id =>
      if fixedMap.contains(id) then (id, fixedMap(id), constraintNoise)
      else (id, icpMap(id), icpNoise)
    }

    val posterior = model.posterior(regressionData)
    posterior.mean

  // -----------------------------
  // Nonrigid ICP loop (keeps external constraints every iteration)
  // -----------------------------
  def nonrigidICP(
      movingMesh: TriangleMesh[_3D],
      ptIds: Seq[PointId],
      iters: Int
  ): TriangleMesh[_3D] =
    if iters <= 0 then movingMesh
    else
      val icpCorr =
        if onlyConstraints then IndexedSeq.empty
        else attributeCorrespondences(movingMesh, ptIds)

      val transformed = fitModel(icpCorr, fixedConstraints)
      nonrigidICP(transformed, ptIds, iters - 1)

  // -----------------------------
  // Run
  // -----------------------------
  val initMesh = model.mean
  val finalFit = nonrigidICP(initMesh, sampledPtIds, nIterations)

  val resultGroup = ui.createGroup("results")
  ui.show(resultGroup, initMesh, "init_mean")
  ui.show(resultGroup, finalFit, "final_fit")

  val outFitPath = new File("output/final_fit_with_constraints.stl")
  MeshIO.writeMesh(finalFit, outFitPath) match
    case Success(_) => println(s"✅ Saved final fit STL: ${outFitPath.getAbsolutePath}")
    case Failure(e) => println(s"❌ Failed to save STL: ${e.getMessage}")
