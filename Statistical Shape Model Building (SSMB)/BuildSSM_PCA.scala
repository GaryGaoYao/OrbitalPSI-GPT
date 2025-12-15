//> using scala "3.3"
//> using dep "ch.unibas.cs.gravis::scalismo-ui:0.92.0"

import scalismo.ui.api.*
import scalismo.geometry.*
import scalismo.common.*
import scalismo.mesh.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.statisticalmodel.*
import scalismo.statisticalmodel.dataset.*

import java.io.File
import scala.util.{Try, Success, Failure}

object BuildSSM_PCA extends App:

  scalismo.initialize()
  implicit val rng: scalismo.utils.Random = scalismo.utils.Random(42)
  val ui = ScalismoUI()

  // -------- Paths (repo-relative) --------
  val datasetDir   = new File("Orbital-Bone-Dataset")
  val referenceF   = new File(datasetDir, "reference-orbital-bone.stl")

  val outDir       = new File("output/ssm_pca")
  if (!outDir.exists()) outDir.mkdirs()

  val outModelH5   = new File(outDir, "ssm_pca.h5")
  val exportDir    = new File(outDir, "stl_exports")
  if (!exportDir.exists()) exportDir.mkdirs()

  // -------- Load reference --------
  val reference: TriangleMesh3D =
    MeshIO.readMesh(referenceF) match
      case Success(m) => m
      case Failure(e) => throw new RuntimeException(s"Failed to read reference: ${e.getMessage}")

  // -------- Load dataset meshes --------
  require(datasetDir.exists() && datasetDir.isDirectory, s"Dataset dir not found: ${datasetDir.getPath}")

  val meshFiles = datasetDir.listFiles()
    .filter(f => f.isFile && f.getName.toLowerCase.endsWith(".stl"))
    .filter(f => f.getCanonicalPath != referenceF.getCanonicalPath)
    .sortBy(_.getName)

  require(meshFiles.nonEmpty, s"No STL meshes found in: ${datasetDir.getPath}")

  val dsGroup = ui.createGroup("datasets")
  val meshes: IndexedSeq[TriangleMesh3D] =
    meshFiles.map { f =>
      val m = MeshIO.readMesh(f).get
      ui.show(dsGroup, m, f.getName)
      m
    }.toIndexedSeq

  // -------- Topology check --------
  meshes.zip(meshFiles).foreach { (m, f) =>
    require(
      m.triangulation == reference.triangulation,
      s"Topology mismatch: ${f.getName} (must match reference triangulation)"
    )
  }
  println(s"✅ Loaded ${meshes.length} meshes. Topology consistent.")

  // -------- Build PCA model --------
  val dc = DataCollection.fromTriangleMesh3DSequence(reference, meshes)
  val model = PointDistributionModel.createUsingPCA(dc)

  println(s"✅ PCA SSM created. rank=${model.rank}")

  // -------- Save (.h5) --------
  StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, outModelH5) match
    case Success(_) => println(s"✅ Model saved: ${outModelH5.getPath}")
    case Failure(e) => throw new RuntimeException(s"Failed to save model: ${e.getMessage}")

  // -------- Export STL --------
  def writeStl(mesh: TriangleMesh3D, name: String): Unit =
    MeshIO.writeMesh(mesh, new File(exportDir, name)) match
      case Success(_) => println(s"✅ STL saved: ${new File(exportDir, name).getPath}")
      case Failure(e) => println(s"❌ STL write failed ($name): ${e.getMessage}")

  writeStl(model.mean, "mean.stl")
  for i <- 1 to 5 do writeStl(model.sample(), f"sample_${i}%03d.stl")

  // -------- Visualize --------
  val g = ui.createGroup("SSM_PCA")
  ui.show(g, model, "PDM (PCA)")
  ui.show(g, model.mean, "Mean")

  println("✅ Done.")
