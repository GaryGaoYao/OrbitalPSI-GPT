//> using scala "3.3"
//> using dep "ch.unibas.cs.gravis::scalismo-ui:0.92.0"

import scalismo.ui.api.*
import scalismo.geometry.*
import scalismo.common.*
import scalismo.mesh.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.statisticalmodel.*
import scalismo.statisticalmodel.dataset.*
import scalismo.kernels.*
import scalismo.numerics.PivotedCholesky.RelativeTolerance

import java.io.File
import scala.util.{Try, Success, Failure}
import scala.math.*

object BuildRegionAwareMultiscaleGPMM extends App:

  // ---------------------------
  // 0) Init
  // ---------------------------
  scalismo.initialize()
  implicit val rng: scalismo.utils.Random = scalismo.utils.Random(42)
  val ui = ScalismoUI()

  // ---------------------------
  // 1) Config
  // ---------------------------
  val datasetDir        = new File(raw"D:\Codes\Scalismo\orbital-bone-aligned-registered")
  val referenceMeshPath = new File(raw"D:\Codes\Scalismo\reference-orbital-bone.stl")

  val outDir            = new File(raw"D:\Codes\Scalismo\output\gpmm_region_aware")
  if (!outDir.exists()) outDir.mkdirs()

  val outModelH5        = new File(outDir, "region_aware_multiscale_gpmm.h5")
  val exportDir         = new File(outDir, "stl_exports")
  if (!exportDir.exists()) exportDir.mkdirs()

  // ---- clustering levels ----
  val levelsK = Seq(2, 3, 4)
  val baseSigmas  = Seq(2.0, 5.0, 10.0, 20.0)   // L scales
  val alphaScales = Seq(1.0, 0.8, 0.5, 0.25)    // weights
  require(baseSigmas.length == alphaScales.length)

  // ---- clinical weighting mapping ----
  val gammaAmp = 1.0
  val betaLen  = 0.5
  val maskSigmaFactor = 1.5

  val globalScale = 10.0 
  val globalSigmaDiv = 4.0   // globalSigma = diagonal / globalSigmaDiv

  // ---- low-rank approx ----
  val relTol = 0.02

  // ---------------------------
  // 2) Load reference + dataset meshes
  // ---------------------------
  val reference: TriangleMesh3D =
    MeshIO.readMesh(referenceMeshPath) match
      case Failure(ex) =>
        println(s"❌ Failed to load reference: ${ex.getMessage}")
        sys.exit(1)
      case Success(m) =>
        println(s"✅ Reference loaded: ${referenceMeshPath.getPath}")
        m

  require(datasetDir.exists() && datasetDir.isDirectory, s"❌ Dataset dir not found: ${datasetDir.getPath}")

  val meshFiles = datasetDir.listFiles()
    .filter(f => f.isFile && f.getName.toLowerCase.endsWith(".stl"))
    .filter(f => f.getCanonicalPath != referenceMeshPath.getCanonicalPath)
    .sortBy(_.getName)

  require(meshFiles.nonEmpty, s"❌ No STL found in: ${datasetDir.getPath}")

  val dsGroup = ui.createGroup("datasets")
  val meshes: IndexedSeq[TriangleMesh3D] =
    meshFiles.map { f =>
      val m = MeshIO.readMesh(f).get
      ui.show(dsGroup, m, f.getName)
      m
    }.toIndexedSeq

  meshes.zip(meshFiles).foreach { (m, f) =>
    require(
      m.triangulation == reference.triangulation,
      s"❌ Topology mismatch: ${f.getName} (must match reference triangulation)"
    )
  }
  println(s"✅ Loaded ${meshes.length} meshes with consistent topology.")

  val alignedMeshes = meshes

  val templatePoints: IndexedSeq[Point[_3D]] = reference.pointSet.points.toIndexedSeq
  val diag = reference.boundingBox.diagonal.norm
  val globalSigma = diag / globalSigmaDiv

  def clinicalScoreOfCenter(c: Point[_3D], bb: BoundingBox[_3D]): Double =
    val zMin = bb.origin.z
    val zMax = bb.oppositeCorner.z
    val xMin = bb.origin.x
    val xMax = bb.oppositeCorner.x
    val zNorm = (c.z - zMin) / max(1e-12, (zMax - zMin)) // 0~1
    val xNorm = (c.x - xMin) / max(1e-12, (xMax - xMin)) // 0~1

    val floorScore    = 1.0
    val superiorScore = 0.2
    val medialScore   = 0.7
    val lateralScore  = 0.4

    // 规则：优先用上下判断 floor/superior
    // 否则用左右判断 medial/lateral
    if (zNorm <= 0.35) floorScore
    else if (zNorm >= 0.65) superiorScore
    else if (xNorm <= 0.5) medialScore
    else lateralScore

  // soft mask：以中心点为核的高斯权重
  def softMask(center: Point[_3D], maskSigma: Double): Point[_3D] => Double =
    (p: Point[_3D]) =>
      exp(-pow((p - center).norm, 2) / (2.0 * maskSigma * maskSigma))

  // 在模板上为每个 level(k) 生成 k 个中心
  val bb = reference.boundingBox
  val allRegionCenters: IndexedSeq[Point[_3D]] =
    levelsK.flatMap { k =>
      rng.scalaRandom.shuffle(templatePoints).take(k)
    }.toIndexedSeq

  println(s"✅ Template regional centers: total=${allRegionCenters.length} (k=2/3/4 -> 9 regions)")

  // ---------------------------
  // 4) Region-aware multiscale kernel (localized RBF, modulated by clinical scores)
  //    K(x,y)= Σ_r  w_r(x) w_r(y) Σ_l alpha_l * exp(-||x-y||^2/(2*sigma_{l,r}^2)) * g(s_r)  * I3
  //    sigma_{l,r} = baseSigma_l * h(s_r) ，h(s)=s^{-beta}
  // ---------------------------

  // helper：空间加权核（标量核）
  case class SpatiallyWeightedKernel(baseKernel: PDKernel[_3D], weightFn: Point[_3D] => Double) extends PDKernel[_3D]:
    override def domain: Domain[_3D] = baseKernel.domain
    override def k(x: Point[_3D], y: Point[_3D]) =
      baseKernel.k(x, y) * (weightFn(x) * weightFn(y))

  // 构造“每个 region 的 multiscale kernel 叠加”
  val maskSigmaBase = diag / (levelsK.max.toDouble * maskSigmaFactor)

  val regionKernels: IndexedSeq[PDKernel[_3D]] =
    allRegionCenters.zipWithIndex.map { (c, idx) =>
      val s = clinicalScoreOfCenter(c, bb) // 1.0 / 0.7 / 0.4 / 0.2
      val amp = pow(s, gammaAmp)           // g(s)=s^gamma
      val lenScaleFactor = pow(s, -betaLen) // h(s)=s^{-beta}：重要区域->更小 sigma

      val wFn = softMask(c, maskSigmaBase)

      // multiscale sum for this region
      val regionalMultiScale: PDKernel[_3D] =
        baseSigmas.zip(alphaScales).map { (sigma0, a) =>
          val sigma = sigma0 * lenScaleFactor
          // amplitude: globalScale * amp * a
          val ker = GaussianKernel3D(sigma) * (globalScale * amp * a)
          ker
        }.reduce(_ + _)

      // localized: w(x) K w(y)
      SpatiallyWeightedKernel(regionalMultiScale, wFn)
    }

  // 全局 kernel
  val globalKernel: PDKernel[_3D] = GaussianKernel3D(globalSigma) * (globalScale * 0.5)

  val combinedKernel: PDKernel[_3D] =
    regionKernels.foldLeft(globalKernel)(_ + _)

  println("✅ Region-aware multiscale kernel constructed.")

  // ---------------------------
  // 5) Build GP + low-rank approximation -> GPMM / PDM
  // ---------------------------

  val meanField = Field(RealSpace3D, (_: Point[_3D]) => Vector3D(0, 0, 0))
  val gp = GaussianProcess(meanField, combinedKernel)

  println(s"--- Approximating GP (Pivoted Cholesky) with relativeTolerance=$relTol ---")
  val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
    reference,
    gp,
    relativeTolerance = relTol,
    interpolator = NearestNeighborInterpolator3D()
  )

  // 形状模型（GPMM / PDM）
  val model = PointDistributionModel(lowRankGP)
  println(s"✅ Model rank = ${model.rank}")

  // ---------------------------
  // 6) Save model (.h5)
  // ---------------------------
  StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, outModelH5) match
    case Success(_) => println(s"✅ Saved model: ${outModelH5.getPath}")
    case Failure(e) =>
      println(s"❌ Failed to save model: ${e.getMessage}")
      sys.exit(1)

  // ---------------------------
  // 7) Export STL (mean + samples)
  // ---------------------------
  def writeStl(mesh: TriangleMesh3D, filename: String): Unit =
    val f = new File(exportDir, filename)
    MeshIO.writeMesh(mesh, f) match
      case Success(_) => println(s"✅ STL saved: ${f.getPath}")
      case Failure(e) => println(s"❌ STL write failed (${f.getName}): ${e.getMessage}")

  writeStl(model.mean, "mean_region_aware_gpmm.stl")

  val nSamples = 10
  for i <- 1 to nSamples do
    writeStl(model.sample(), f"sample_${i}%03d.stl")

  // ---------------------------
  // 8) Visualize
  // ---------------------------
  val g = ui.createGroup("RegionAwareMultiscaleGPMM")
  ui.show(g, reference, "Template(Reference)")
  ui.show(g, model, "PDM (Region-aware multiscale GPMM)")
  ui.show(g, model.mean, "Mean")

  println("✅ Done.")
