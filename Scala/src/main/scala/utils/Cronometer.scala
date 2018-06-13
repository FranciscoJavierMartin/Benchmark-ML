package utils

class Cronometer (val start_time:Long=System.currentTimeMillis()) {

  def printTime():Unit={
    val total_time=System.currentTimeMillis()-start_time
    println(s"Training time: ${total_time} ms")
    println(s"Training time: ${total_time/3600000} hours, ${(total_time/60000)%60} minutes, ${(total_time/1000)%60} seconds")
  }

  def appendTime(filename:String,args:Array[String]):Unit={
    val total_time=System.currentTimeMillis()-start_time
    scala.tools.nsc.io.File(filename).appendAll(s"Scala AppName: ${args(0)} Master: ${args(1)} Time: ${total_time} milliseconds\n")
    scala.tools.nsc.io.File(filename).appendAll(s"Training time: ${total_time/3600000} hours, ${(total_time/60000)%60} minutes, ${(total_time/1000)%60} seconds\n")
  }
}