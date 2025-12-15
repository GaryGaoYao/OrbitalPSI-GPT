//> using scala "3.3.1"
//> using repository "https://scalismo.org/repository"
//> using dep "ch.unibas.cs.gravis::scalismo:0.92.1"

import scalismo.io._
import scalismo.geometry._
import scalismo.registration._
import scalismo.utils.Random
import java.io.File



@main def batchAlign() =
  scalismo.initialize()
  implicit val rng: Random = Random(42)

  val refMesh = MeshIO.readMesh(new File("reference-orbital-bone.stl")).get
  val refLms = LandmarkIO.readLandmarksJson[_3D](new File("reference-orbital-bone.json")).get

  val rawFolder = new File("orbital-bone/")
  val alignedFolder = new File("orbital-bone-aligned/")
  alignedFolder.mkdirs()

  val plyFiles = rawFolder.listFiles().filter(f => f.getName.endsWith(".stl"))

  plyFiles.foreach { plyFile =>
    val name = plyFile.getName.stripSuffix(".stl")
    val jsonFile = new File(rawFolder, name + ".json")

    if (jsonFile.exists()) {
      println(s"Processing $name")

      val mesh = MeshIO.readMesh(plyFile).get
      val lms = LandmarkIO.readLandmarksJson[_3D](jsonFile).get

      val transform = LandmarkRegistration.rigid3DLandmarkRegistration(lms, refLms, Point(0, 0, 0))
      val alignedMesh = mesh.transform(transform)
      val alignedLms = lms.map(lm => lm.copy(point = transform(lm.point)))

      MeshIO.writeMesh(alignedMesh, new File(alignedFolder, s"${name}_aligned.stl"))
      //LandmarkIO.writeLandmarksJson[_3D](alignedLms, new File(alignedFolder, s"${name}_aligned.json"))
    } else {
      println(s"WARNING: JSON file not found for $name")
    }
  }

  println("done")
