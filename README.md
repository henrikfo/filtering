# filtering
This work is a PoC-merge of ColonyOS(https://github.com/colonyos), Digital Earth Sweden(https://github.com/DigitalEarthSweden) and cloud detection models(https://github.com/aleksispi/ml-cloud-opt-thick).
Use trained models to filter satelite images based on cloudiness and then download the cloud-free ones.  

Prerequisites include docker, docker-compose and ColonyOS.
To understand each individual part better please check those repos out.

Future use-cases building ontop of this PoC:  
Utilize ColonyOS to process large amounts of satelite data for various use cases, e.g. dataset creation.  

Refine the PoC and make the solution more intuative.  
Incorporate some interactive element such as a map, fetch multiple coordinates(area boxes) to a collection, and set a timeframe for the whole collection, and then download it all.  
  
Continuous updating map in ”real-time".  
Take inspiration of ColonyOS and from this PoC, develop and release a product that update a, cloud-free, map of sweden every single day. Partition work over multiple executors for fast processing. Update grind-like map in a masked based approach  
  
Build pipeplines for collection of large data for (un)supervised learning.  
Utilize the  flexible nature of ColonyOS workflows and setup complex pipelines for data collection, processing abd gathering of large amounts of data.

DEPLOY OTHER USE-CASES WITH PRE-TRAINED MODELS FOR EVENT DETECTION/ALARM 
Deploy usecases ontop of this PoC.
Using a set of different pretrained models, configure an event detection/alarm system. Get notified when a particular event happens.  
Can be spatially specific or not, meaning you might be interested in a particular area or event. Can be continuously ran, indefinite.  



