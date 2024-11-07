import traci

sumo_binary = "sumo-gui" 
sumo_config_file = "./Simulation/network.sumocfg" 

traci.start([sumo_binary, "-c", sumo_config_file])

try:
    step = 0
    while step < 100:  
        traci.simulationStep() 


        vehicles = traci.vehicle.getIDList()
        print("Pas de temps", step, "- Véhicules :", vehicles)

        if not vehicles:
            print("Aucun véhicule chargé. Vérifiez le fichier .rou.xml")

        step += 1

finally:
    traci.close()  
