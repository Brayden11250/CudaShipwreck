# Compiling and running
Reverses Structure Seeds from a shipwreck's Position, Rotation, and Type.


Compile:
  
         nvcc -o shipwreck_bruteforcer shipwreck_bruteforcer.cu -O3 -std=c++17
         
Run:
         ./shipwreck_bruteforcer {textfilename}.txt

Format for text file:

{ChunkX},{ChunkZ},{Rotation},{Type},{Ocean or Beached}

Example inputs:

17,-23,CLOCKWISE_90,with_mast_degraded,Ocean

-16,-21,COUNTERCLOCKWISE_90,rightsideup_fronthalf,Beached


# Filtering
Runs well with 2+ shipwrecks

2 Shipwrecks gives ~450000 Stucture Seeds (lowest I've seen is 260504 highest I've seen is 574771 though most of those tests didn't include beached)

3 Shipwrecks gives a few dozen at most.

# Why?

Ocean: 1 in 80 (4 rotations × 20 types)

Beached: 1 in 44 (4 rotations × 11 types)

These probabilities multiply with each added shipwreck

If you had three shipwrecks 2 ocean and 1 beached that would mean that only 1/80x80x44 (1/281600) of the structure seeds that are capable of spawning a shipwrecks at ***ALL*** three points would pass leading to a double or very low triple digit answer
