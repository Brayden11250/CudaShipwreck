# CudaShipwreck



  • Compile using
  
         nvcc -o shipwreck_bruteforcer shipwreck_bruteforcer.cu -O3 -std=c++17
         
  • Run
         ./shipwreck_bruteforcer {textfilename}.txt

Format:
{ChunkX},{ChunkZ},{Rotation},{type},{Ocean or Beached}

Example:



17,-23,CLOCKWISE_90,with_mast_degraded,Ocean
-16,-21,COUNTERCLOCKWISE_90,rightsideup_fronthalf,Ocean


Runs well with 2+ shipwrecks behaviour with 1 shipwreck hasn't been tested.
