#
#  ISC License
#
#  Copyright (c) 2022, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

r"""
Overview
--------

This script shows how to perform momentum dumping when the momentum accumulated on the reaction wheels
is above a user-defined threshold. In this case, such threshold is set at 80 Nms. The dumping is performed 
by a set of 8 thrusters that can provide control about the three principal axes of the spacecraft. 
To perform the momentum dumping, three concatenated modules are used:

- :ref:`thrMomentumManagement`: computes the amount of momentum to be dumped, based on current stored momentum 
  and the user-defined threshold. It is important to notice that, for the three concatenated modules to work
  correctly, this first module cannot be run at simulation time :math:`t = 0`. In this script, the method 
  ``Reset`` is called on :ref:`thrMomentumManagement` at :math:`t = 10` s, which coincides to the time at which 
  the first desaturating impulse is fired.
- :ref:`thrForceMapping`: maps the amout of momentum to be dumped into impulses that must be delivered by each
  thruster. This module is originally implemented to map a requested torque into forces imparted by the thrusters,
  but it can be applied in this case as well, because the math is the same. The only caveat is that, in this case,
  the output should not be scaled by the thruster maximum torque capability, since the desired output is an impulse
  and not a torque. To deactivate the output scaling, the ``angErrThresh`` input variable for this module must be 
  set to a value larger than :math:`\pi`, as specified in the module documentation.
- :ref:`thrMomentumDumping`: computes the thruster on-times required to deliver the desired impulse. A 
  ``maxCounterValue`` of 100 is used in this example to allow the spacecraft to maneuver back to the desired attitude
  after each time the thrusters fire. 

For this script to work as intended, it is necessary to run the flight software and the dynamics at two different 
frequencies. In this example, the simulation time step for the flight software is 1 second, whereas for the dynamics
it is 0.1 seconds. This is necessary because the :ref:`thrMomentumDumping` automatically uses the task time step as 
control period for the firing. However, if the dynamics is integrated at the same frequency, this does not give 
enough time resolution to appreciate the variation in the momentum.

The script is found in the folder ``basilisk/examples`` and executed by using::

      python3 scenarioMomentumDumping.py

Illustration of Simulation Results
----------------------------------

In this examples, the spacecraft is already at the desired attitude, but the four reaction wheels are saturated (the total
angular momentum exceeds the threshold). The desaturation happens at :math:`t = 10` when the :ref:`thrMomentumManagement` is
reset. Three firings are sufficient to dump the momentum below the set threshold. The following figures illustrate the change
in momentum for the four wheels :math:`H_i` for :math:`i = 1,...,4` and the total angular momentum :math:`\|H\|`, and the 
attitude errors, as functions of time, with respect to the desired target attitude.

.. image:: /_images/Scenarios/scenarioMomentumDumping3.svg
   :align: center

.. image:: /_images/Scenarios/scenarioMomentumDumping1.svg
   :align: center

The plots show that the momentum is dumped below the threshold. Also, the desired attitude is recovered between the first and
second firing, and after the third, but between the second and the third there is not enough time for the spacecraft to slew
back to that attitude.

The next two plots show the amount of impulse [Ns] requested for each thruster, and the times during which each thruster is 
operational. As expected, 100 control times pass between each firing: because the control time coincides with the flight 
software simulation time step of 1 s, this means that firings are 100 seconds apart.

.. image:: /_images/Scenarios/scenarioMomentumDumping5.svg
   :align: center

.. image:: /_images/Scenarios/scenarioMomentumDumping7.svg
   :align: center

"""

import os

import Basilisk.architecture.cMsgCInterfacePy as cMsgPy
import matplotlib.pyplot as plt
import numpy as np
from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import (mrpFeedback, attTrackingError, oneAxisSolarArrayPoint, rwMotorTorque,
                                    hingedRigidBodyPIDMotor, solarArrayReference, thrusterPlatformReference,
                                    thrusterPlatformState, thrustCMEstimation, torqueScheduler)
from Basilisk.simulation import (reactionWheelStateEffector, simpleNav, simpleMassProps, spacecraft,
                                 spinningBodyOneDOFStateEffector,
                                 spinningBodyTwoDOFStateEffector, thrusterStateEffector, facetSRPDynamicEffector)
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody, simIncludeRW,
                                unitTestSupport, vizSupport, RigidBodyKinematics as rbk)

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])



def run(show_plots):

    # Create simulation variable names
    fswTask = "fswTask"
    pltRefTask = "pltRefTask"
    dynTask = "dynTask"
    simProcessName = "simProcess"

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    scSim.SetProgressBar(True)

    #  create the simulation process
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # create the dynamics task and specify the simulation time and integration update time
    simulationTime = macros.day2nano(2)
    simulationTimeStepDyn = macros.sec2nano(0.1)
    simulationTimeStepFsw = macros.sec2nano(1)
    simulationTimeStepPlt = macros.hour2nano(1)
    dynProcess.addTask(scSim.CreateNewTask(dynTask, simulationTimeStepDyn))
    dynProcess.addTask(scSim.CreateNewTask(pltRefTask, simulationTimeStepPlt))
    dynProcess.addTask(scSim.CreateNewTask(fswTask, simulationTimeStepFsw))
    
    #
    # setup the simulation tasks/objects
    # 

    # initialize spacecraft object and set properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "Spacecraft"

    # add spacecraft object to the simulation process
    scSim.AddModelToTask(dynTask, scObject, 1)

    # setup Gravity Body
    gravFactory = simIncludeGravBody.gravBodyFactory()

    # Next a series of gravitational bodies are included
    gravBodies = gravFactory.createBodies(['sun'])
    gravBodies['sun'].isCentralBody = True
    mu = gravBodies['sun'].mu
    
    # The configured gravitational bodies are added to the spacecraft dynamics with the usual command:
    scObject.gravField.gravBodies = spacecraft.GravBodyVector(list(gravFactory.gravBodies.values()))

    # Next, the default SPICE support module is created and configured.
    timeInitString = "2023 OCTOBER 22 00:00:00.0"

    # The following is a support macro that creates a `gravFactory.spiceObject` instance
    gravFactory.createSpiceInterface(bskPath +'/supportData/EphemerisData/',
                                     timeInitString,
                                     epochInMsg=True)

    # Earth is gravity center
    gravFactory.spiceObject.zeroBase = 'Sun'

    # The SPICE object is added to the simulation task list.
    scSim.AddModelToTask(fswTask, gravFactory.spiceObject, 2)

    # The gravitational body is connected to the spacecraft object
    scObject.gravField.gravBodies = spacecraft.GravBodyVector(list(gravFactory.gravBodies.values()))

    # setup the orbit using classical orbit elements
    oe = orbitalMotion.ClassicElements()
    oe.a = 100e9      # meters
    oe.e = 0.001
    oe.i = 0.0 * macros.D2R
    oe.Omega = 0.0 * macros.D2R
    oe.omega = 0.0 * macros.D2R
    oe.f = -135.0 * macros.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe)

    # To set the spacecraft initial conditions, the following initial position and velocity variables are set:
    scObject.hub.r_CN_NInit = rN                          # m   - r_BN_N
    scObject.hub.v_CN_NInit = vN                          # m/s - v_BN_N
    scObject.hub.sigma_BNInit = [0, 0., 0.]              # MRP set to customize initial inertial attitude
    scObject.hub.omega_BN_BInit = [[0.], [0.], [0.]]      # rad/s - omega_CN_B
    
    # define the simulation inertia
    I = [ 1725,    -5,   -12,
            -5,  5525,    43,
            -12,   43,  4810]
    scObject.hub.mHub = 2500  # kg - spacecraft mass
    scObject.hub.r_BcB_B = [[0.008], [-0.010], [1.214]]  # [m] - position vector of hub CM relative to the body-fixed point B
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

    #
    # add RW devices
    #
    # Make RW factory instance
    rwFactory = simIncludeRW.rwFactory()

    # store the RW dynamical model type
    varRWModel = messaging.BalancedWheels

    # specify RW momentum capacity
    maxRWMomentum = 100.  # Nms

    # Define orthogonal RW pyramid
    # -- Pointing directions
    rwElAngle = np.array([40.0, 40.0, 40.0, 40.0]) * macros.D2R
    rwAzimuthAngle = np.array([45.0, 135.0, 225.0, 315.0]) * macros.D2R
    rwPosVector = [[0.8, 0.8, 1.79070],
                    [0.8, -0.8, 1.79070],
                    [-0.8, -0.8, 1.79070],
                    [-0.8, 0.8, 1.79070]]

    Gs = []
    for elAngle, azAngle, posVector in zip(rwElAngle, rwAzimuthAngle, rwPosVector):
        gsHat = (rbk.Mi(-azAngle, 3).dot(rbk.Mi(elAngle, 2))).dot(np.array([1, 0, 0]))
        Gs.append(gsHat)
        rwFactory.create('Honeywell_HR16', gsHat, maxMomentum=maxRWMomentum, rWB_B=posVector, Omega=0.)

    numRW = rwFactory.getNumOfDevices()

    # create RW object container and tie to spacecraft object
    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwStateEffector.ModelTag = "RW_cluster"
    rwFactory.addToSpacecraft(scObject.ModelTag, rwStateEffector, scObject)

    # add RW object array to the simulation process
    scSim.AddModelToTask(dynTask, rwStateEffector, 2)

    # Setup the FSW RW configuration message.
    fswRwConfigMsg = rwFactory.getConfigMessage()

    # add the simple Navigation sensor module
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(dynTask, sNavObject)
    
    # Set up the simple mass props object
    simpleMassPropsObject = simpleMassProps.SimpleMassProps()
    scSim.AddModelToTask(dynTask, simpleMassPropsObject)
    
    # Set up the rotating solar arrays
    numRSA = 2
    RSAList = []
    # 1st solar array
    RSAList.append(spinningBodyOneDOFStateEffector.SpinningBodyOneDOFStateEffector())
    scSim.AddModelToTask(dynTask, RSAList[0])
    RSAList[0].r_SB_B = [0.75, 0.0, 0.45]
    RSAList[0].r_ScS_S = [0.0, 3.75, 0.0]
    RSAList[0].sHat_S = [0, 1, 0]
    RSAList[0].dcm_S0B = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
    RSAList[0].IPntSc_S = [[250.0, 0.0, 0.0],
                           [0.0, 250.0, 0.0],
                           [0.0, 0.0, 500.0]]
    RSAList[0].mass = 85
    RSAList[0].k = 0
    RSAList[0].c = 0
    RSAList[0].thetaInit = 0
    RSAList[0].thetaDotInit = 0
    RSAList[0].ModelTag = "solarArray1"
    scObject.addStateEffector(RSAList[0])
    # 2nd solar array
    RSAList.append(spinningBodyOneDOFStateEffector.SpinningBodyOneDOFStateEffector())
    scSim.AddModelToTask(dynTask, RSAList[1])
    RSAList[1].r_SB_B = [-0.75, 0.0, 0.45]
    RSAList[1].r_ScS_S = [0.0, 3.75, 0.0]
    RSAList[1].sHat_S = [0, 1, 0]
    RSAList[1].dcm_S0B = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    RSAList[1].IPntSc_S = [[250.0, 0.0, 0.0],
                           [0.0, 250.0, 0.0],
                           [0.0, 0.0, 500.0]]
    RSAList[1].mass = 85
    RSAList[1].k = 0
    RSAList[1].c = 0
    RSAList[1].thetaInit = 0
    RSAList[1].thetaDotInit = 0
    RSAList[1].ModelTag = "solarArray2"
    scObject.addStateEffector(RSAList[1])
    
    # Set up the dual-gimbaled platform
    platform = spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector()
    scSim.AddModelToTask(dynTask, platform)
    platform.theta1Init = 0
    platform.theta1DotInit = 0
    platform.theta2Init = 0
    platform.theta2DotInit = 0
    platform.mass1 = 0
    platform.mass2 = 10
    platform.k1 = 0
    platform.k2 = 0
    platform.r_S1B_B = [0, 0, 0]
    platform.r_S2S1_S1 = [0, 0, 0]
    platform.r_Sc1S1_S1 = [0, 0, 0]
    platform.r_Sc2S2_S2 = [0, 0, 0]
    platform.s1Hat_S1 = [1, 0, 0]
    platform.s2Hat_S2 = [0, 1, 0]
    platform.IS1PntSc1_S1 = [[2, 0, 0], [0, 3, 0], [0, 0, 4]]
    platform.IS2PntSc2_S2 = [[2, 0, 0], [0, 3, 0], [0, 0, 4]]
    platform.dcm_S10B = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    platform.dcm_S20S1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    platform.ModelTag = "platform1"
    scObject.addStateEffector(platform)
    
    # Set up the SEP thruster
    sepThruster = thrusterStateEffector.ThrusterStateEffector()
    scSim.AddModelToTask(dynTask, sepThruster)
    thruster = thrusterStateEffector.THRSimConfig()
    r_TF_F = [0, 0, 0]  # Thruster application point in F frame coordinates
    tHat_F = [0, 0, 1]  # Thrust unit direction vector in F frame coordinates
    thruster.thrLoc_B = r_TF_F
    thruster.thrDir_B = tHat_F
    thruster.MaxThrust = 0.27
    thruster.steadyIsp = 1600
    thruster.MinOnTime = 0.006
    thruster.cutoffFrequency = 5

    sepThruster.addThruster(thruster, platform.spinningBodyConfigLogOutMsgs[1])
    sepThruster.kappaInit = messaging.DoubleVector([0.0])
    sepThruster.ModelTag = "sepThruster"
    scObject.addStateEffector(sepThruster)
    
    # Write THR Config Msg with initial best estimates
    THRConfig = messaging.THRConfigMsgPayload()
    THRConfig.rThrust_B = r_TF_F
    THRConfig.tHatThrust_B = tHat_F
    THRConfig.maxThrust = thruster.MaxThrust
    thrConfigFMsg = messaging.THRConfigMsg().write(THRConfig)
    
    # Set up the SRP dynamic effector
    SRP = facetSRPDynamicEffector.FacetSRPDynamicEffector()
    scSim.AddModelToTask(dynTask, SRP)
    # Define the spacecraft geometry for populating the FacetedSRPSpacecraftGeometryData structure in the SRP module
    # Define the facet surface areas
    lenXHub = 1.53  # [m]
    lenYHub = 1.8  # [m]
    lenZHub = 2.86  # [m]
    area2 = np.pi*(0.5 * 7.262)*(0.5 * 7.262)  # [m^2]
    facetAreas = [lenYHub * lenZHub, lenXHub * lenZHub, lenYHub * lenZHub, lenXHub * lenZHub, lenXHub * lenYHub, lenXHub * lenYHub, area2, area2, area2, area2]

    # Define the facet normals in B frame components
    facetNormal1 = np.array([1.0, 0.0, 0.0])
    facetNormal2 = np.array([0.0, 1.0, 0.0])
    facetNormal3 = np.array([-1.0, 0.0, 0.0])
    facetNormal4 = np.array([0.0, -1.0, 0.0])
    facetNormal5 = np.array([0.0, 0.0, 1.0])
    facetNormal6 = np.array([0.0, 0.0, -1.0])
    facetNormal7 = np.array([0.0, 1.0, 0.0])
    facetNormal8 = np.array([0.0, -1.0, 0.0])
    facetNormal9 = np.array([0.0, 1.0, 0.0])
    facetNormal10 = np.array([0.0, -1.0, 0.0])
    normals_B = [facetNormal1, facetNormal2, facetNormal3, facetNormal4, facetNormal5, facetNormal6, facetNormal7, facetNormal8, facetNormal9, facetNormal10]

    # Define the facet center of pressure locations with respect to point B in B frame components
    facetLoc1 = np.array([0.5 * lenXHub, 0.0, 0.5 * lenZHub])  # [m]
    facetLoc2 = np.array([0.0, 0.5 * lenYHub, 0.5 * lenZHub])  # [m]
    facetLoc3 = np.array([-0.5 * lenXHub, 0.0, 0.5 * lenZHub])  # [m]
    facetLoc4 = np.array([0.0, -0.5 * lenYHub, 0.5 * lenZHub])  # [m]
    facetLoc5 = np.array([0.0, 0.0, lenZHub])  # [m]
    facetLoc6 = np.array([0.0, 0.0, 0.0])  # [m]
    facetLoc7 = np.array([3.75 + 0.5 * lenXHub, 0.544, 0.44])  # [m]
    facetLoc8 = np.array([3.75 + 0.5 * lenXHub, 0.544, 0.44])  # [m]
    facetLoc9 = np.array([-(3.75 + 0.5 * lenXHub), 0.544, 0.44])  # [m]
    facetLoc10 = np.array([-(3.75 + 0.5 * lenXHub), 0.544, 0.44])  # [m]
    locationsPntB_B = [facetLoc1, facetLoc2, facetLoc3, facetLoc4, facetLoc5, facetLoc6, facetLoc7, facetLoc8, facetLoc9, facetLoc10]

    # Define the facet optical coefficients
    specCoeff = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    diffCoeff = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    # Populate the scGeometry structure with the facet information
    for i in range(len(facetAreas)):
        SRP.addFacet(facetAreas[i], specCoeff[i], diffCoeff[i], normals_B[i], locationsPntB_B[i])

    SRP.ModelTag = "FacetSRP"
    scObject.addDynamicEffector(SRP)

    #
    #   setup the FSW algorithm tasks
    #
    
    # Set up thruster platform state module
    pltState = thrusterPlatformState.thrusterPlatformState()
    pltState.ModelTag = "thrusterPlatformState"
    pltState.sigma_MB = np.array([0, 0, 0])
    pltState.r_BM_M = [0, 0, 0]
    pltState.r_FM_F = [0, 0, 0]
    scSim.AddModelToTask(fswTask, pltState, 30)
    
    # Set up the CM estimator module
    r_CB_B_0 = [0.04, -0.05, 1.25]
    cmEstimator = thrustCMEstimation.ThrustCMEstimation()
    cmEstimator.ModelTag = "cmEstimator"
    cmEstimator.attitudeTol = 1e-6
    cmEstimator.r_CB_B = r_CB_B_0 # Real CoM_B location = [0.113244, 0.025605, 1.239834]
    cmEstimator.P0 = [0.0025, 0.0025, 0.0025]
    cmEstimator.R0 = [4e-10, 4e-10, 4e-10]
    scSim.AddModelToTask(fswTask, cmEstimator, None, 29)

    # create the FSW vehicle configuration message for inertias
    vehicleConfigData = messaging.VehicleConfigMsgPayload()
    vehicleConfigData.CoM_B = r_CB_B_0    # use the same initial CoM guess as the cmEstimator module
    vcMsg_CoM = messaging.VehicleConfigMsg_C()
    vcMsg_CoM.write(vehicleConfigData)

    # create the FSW vehicle configuration message for inertias
    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    vehicleConfigOut.ISCPntB_B = I       # use the same inertia in the FSW algorithm as in the simulation
    vcMsg_I = messaging.VehicleConfigMsg().write(vehicleConfigOut)
    
    # Set up platform reference module
    pltReference = thrusterPlatformReference.thrusterPlatformReference()
    pltReference.ModelTag = 'thrusterPlatformReference'
    pltReference.sigma_MB = pltState.sigma_MB
    pltReference.r_BM_M = pltState.r_BM_M
    pltReference.r_FM_F = pltState.r_FM_F
    pltReference.theta1Max = np.pi/12
    pltReference.theta2Max = np.pi/12
    pltReference.K = 4e-4
    pltReference.Ki = 0
    scSim.AddModelToTask(pltRefTask, pltReference, 28)
    
    pltController = []
    for item in range(2):
        pltController.append(hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor())
        pltController[item].ModelTag = "PltMototorGimbal"+str(item+1)
        pltController[item].K = 0.5
        pltController[item].P = 3
        scSim.AddModelToTask(fswTask, pltController[item], 27)
    
    # Set up the torque scheduler module
    pltTorqueScheduler = torqueScheduler.torqueScheduler()
    pltTorqueScheduler.ModelTag = "TorqueScheduler"
    pltTorqueScheduler.tSwitch = 60
    pltTorqueScheduler.lockFlag = 0
    scSim.AddModelToTask(fswTask, pltTorqueScheduler, 26)

    # Set up attitude guidance module
    sepPoint = oneAxisSolarArrayPoint.oneAxisSolarArrayPoint()
    sepPoint.ModelTag = "sepPointGuidance"
    sepPoint.a1Hat_B = [1, 0, 0]          # solar array drive axis
    sepPoint.a2Hat_B = [0, 1, 0]          # antiparallel direction to the sensitive surface
    sepPoint.hHat_N = [1, 0, 0]           # random inertial thrust direction
    scSim.AddModelToTask(fswTask, sepPoint, 25)
    
    # Set up the solar array reference modules
    saReference = []
    for item in range(numRSA):
        saReference.append(solarArrayReference.solarArrayReference())
        saReference[item].ModelTag = "SolarArrayReference"+str(item+1)
        saReference[item].a1Hat_B = [(-1)**item, 0, 0]
        saReference[item].a2Hat_B = [0, 1, 0]
        scSim.AddModelToTask(fswTask, saReference[item], 24)

    # Set up solar array controller modules
    saController = []
    for item in range(numRSA):
        saController.append(hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor())
        saController[item].ModelTag = "SolarArrayMotor"+str(item+1)
        saController[item].K = 1.25
        saController[item].P = 50
        saController[item].I = 3e-3
        scSim.AddModelToTask(fswTask, saController[item], 23)
    
    # Set up attitude tracking error
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "AttitudeTrackingError"
    scSim.AddModelToTask(fswTask, attError, 22)
    
    # setup the MRP Feedback control module
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    mrpControl.Ki = 1e-5
    mrpControl.P = 276.24
    mrpControl.K = 8.84
    mrpControl.integralLimit = 2. / mrpControl.Ki * 0.1
    scSim.AddModelToTask(fswTask, mrpControl, 21)

    # add module that maps the Lr control torque into the RW motor torques
    rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
    rwMotorTorqueObj.ModelTag = "rwMotorTorque"
    rwMotorTorqueObj.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    scSim.AddModelToTask(fswTask, rwMotorTorqueObj, 20)

    # Configure thruster on-time message
    thrOnTimeMsgData = messaging.THRArrayOnTimeCmdMsgPayload()
    thrOnTimeMsgData.OnTimeRequest = [3600*24*7]
    thrOnTimeMsg = messaging.THRArrayOnTimeCmdMsg().write(thrOnTimeMsgData)

    # Write cmEstimator output msg to the standalone message vcMsg_CoM
    # This is needed because platformReference runs on its own task at a different frequency,
    # but it receives inputs and provides outputs to modules that run on the main flight software task
    cMsgPy.VehicleConfigMsg_C_addAuthor(cmEstimator.vehConfigOutMsgC, vcMsg_CoM)

    # Connect messages
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    sNavObject.sunStateInMsg.subscribeTo(gravFactory.spiceObject.planetStateOutMsgs[0])
    simpleMassPropsObject.scMassPropsInMsg.subscribeTo(scObject.scMassOutMsg)
    RSAList[0].motorTorqueInMsg.subscribeTo(saController[0].motorTorqueOutMsg)
    RSAList[1].motorTorqueInMsg.subscribeTo(saController[1].motorTorqueOutMsg)
    platform.motorTorqueInMsg.subscribeTo(pltTorqueScheduler.motorTorqueOutMsg)
    platform.motorLockInMsg.subscribeTo(pltTorqueScheduler.effectorLockOutMsg)
    SRP.sunInMsg.subscribeTo(gravFactory.spiceObject.planetStateOutMsgs[0])
    pltState.thrusterConfigFInMsg.subscribeTo(thrConfigFMsg)
    pltState.hingedRigidBody1InMsg.subscribeTo(platform.spinningBodyOutMsgs[0])
    pltState.hingedRigidBody2InMsg.subscribeTo(platform.spinningBodyOutMsgs[1])
    cmEstimator.thrusterConfigBInMsg.subscribeTo(pltState.thrusterConfigBOutMsg)
    cmEstimator.intFeedbackTorqueInMsg.subscribeTo(mrpControl.intFeedbackTorqueOutMsg)
    cmEstimator.attGuidInMsg.subscribeTo(attError.attGuidOutMsg)
    cmEstimator.vehConfigInMsg.subscribeTo(simpleMassPropsObject.vehicleConfigOutMsg)
    # pltReference.vehConfigInMsg.subscribeTo(simpleMassPropsObject.vehicleConfigOutMsg)
    pltReference.vehConfigInMsg.subscribeTo(vcMsg_CoM)
    pltReference.thrusterConfigFInMsg.subscribeTo(thrConfigFMsg)
    pltReference.rwConfigDataInMsg.subscribeTo(fswRwConfigMsg)
    pltReference.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)
    pltTorqueScheduler.motorTorque1InMsg.subscribeTo(pltController[0].motorTorqueOutMsg)
    pltTorqueScheduler.motorTorque2InMsg.subscribeTo(pltController[1].motorTorqueOutMsg)
    sepPoint.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    sepPoint.bodyHeadingInMsg.subscribeTo(pltReference.bodyHeadingOutMsg)
    attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    attError.attRefInMsg.subscribeTo(sepPoint.attRefOutMsg)
    mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    mrpControl.vehConfigInMsg.subscribeTo(vcMsg_I)
    mrpControl.rwParamsInMsg.subscribeTo(fswRwConfigMsg)
    mrpControl.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)
    rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwConfigMsg)
    rwMotorTorqueObj.vehControlInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwMotorTorqueObj.rwMotorTorqueOutMsg)
    for item in range(numRSA):
        saReference[item].attNavInMsg.subscribeTo(sNavObject.attOutMsg)
        saReference[item].attRefInMsg.subscribeTo(sepPoint.attRefOutMsg)
        saReference[item].hingedRigidBodyInMsg.subscribeTo(RSAList[item].spinningBodyOutMsg)
        saController[item].hingedRigidBodyInMsg.subscribeTo(RSAList[item].spinningBodyOutMsg)
        saController[item].hingedRigidBodyRefInMsg.subscribeTo(saReference[item].hingedRigidBodyRefOutMsg)
    for item in range(2):
        pltController[item].hingedRigidBodyInMsg.subscribeTo(platform.spinningBodyOutMsgs[item])
    pltController[0].hingedRigidBodyRefInMsg.subscribeTo(pltReference.hingedRigidBodyRef1OutMsg)
    pltController[1].hingedRigidBodyRefInMsg.subscribeTo(pltReference.hingedRigidBodyRef2OutMsg)
    sepThruster.cmdsInMsg.subscribeTo(thrOnTimeMsg)


    # if this scenario is to interface with the BSK Viz, uncomment the following lines
    # viz = vizSupport.enableUnityVisualization(scSim, dynTask, scObject
    #                                           # , saveFile=fileName
    #                                           , rwEffectorList=rwStateEffector
    #                                           )
    # vizSupport.setActuatorGuiSetting(viz, viewRWPanel=True,
    #                                  viewRWHUD=True,
    #                                  viewThrusterPanel=True,
    #                                  viewThrusterHUD=True
    #                                  )


    #
    #   Setup data logging before the simulation is initialized
    #
    numDataPoints = simulationTime / simulationTimeStepFsw
    samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStepFsw, numDataPoints)

    vehConfigLog = simpleMassPropsObject.vehicleConfigOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(dynTask, vehConfigLog)
    snTransLog = sNavObject.transOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(dynTask, snTransLog)
    snAttLog = sNavObject.attOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(dynTask, snAttLog)
    attErrorLog = attError.attGuidOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(dynTask, attErrorLog)
    attRefLog = sepPoint.attRefOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(dynTask, attRefLog)
    rwMotorLog = rwMotorTorqueObj.rwMotorTorqueOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(dynTask, rwMotorLog)
    rwSpeedLog = rwStateEffector.rwSpeedOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(dynTask, rwSpeedLog)
    thrLog = sepThruster.thrusterOutMsgs[0].recorder(samplingTime)
    scSim.AddModelToTask(dynTask, thrLog)
    cmEstLogs = cmEstimator.cmEstDataOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(dynTask, cmEstLogs)

    # A message is created that stores an array of the Omega wheel speeds
    rwLogs = []
    for item in range(numRW):
        rwLogs.append(rwStateEffector.rwOutMsgs[item].recorder(samplingTime))
        scSim.AddModelToTask(dynTask, rwLogs[item])

    saAngleLogs = []
    saRefAngleLogs = []
    for item in range(numRSA):
        saAngleLogs.append(RSAList[item].spinningBodyOutMsg.recorder(samplingTime))
        scSim.AddModelToTask(dynTask, saAngleLogs[item])
        saRefAngleLogs.append(saReference[item].hingedRigidBodyRefOutMsg.recorder(samplingTime))
        scSim.AddModelToTask(dynTask, saRefAngleLogs[item])

    pltAngleLogs = []
    pltRefAngleLogs = []
    pltRefAngleLogs.append(pltReference.hingedRigidBodyRef1OutMsg.recorder(samplingTime))
    pltRefAngleLogs.append(pltReference.hingedRigidBodyRef2OutMsg.recorder(samplingTime))
    for item in range(2):
        scSim.AddModelToTask(dynTask, pltRefAngleLogs[item])
        pltAngleLogs.append(platform.spinningBodyOutMsgs[item].recorder(samplingTime))
        scSim.AddModelToTask(dynTask, pltAngleLogs[item])

    # initialize Simulation:  This function runs the self_init()
    # cross_init() and reset() routines on each module.
    scSim.InitializeSimulation()

    # configure a simulation stop time and execute the simulation run
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    # retrieve the logged data

    dataSigmaBN = snAttLog.sigma_BN
    dataSigmaRN = attRefLog.sigma_RN
    dataSigmaBR = attErrorLog.sigma_BR
    dataOmegaRW = rwSpeedLog.wheelSpeeds
    dataRealCM = vehConfigLog.CoM_B
    dataStateError = cmEstLogs.stateError
    dataCovariance = cmEstLogs.covariance
    dataPreFit = cmEstLogs.preFitRes
    dataPostFit = cmEstLogs.postFitRes

    dataRW = []
    for i in range(numRW):
        dataRW.append(rwLogs[i].u_current)

    dataNu = []
    dataNuRef = []
    for item in range(2):
        dataNu.append(pltAngleLogs[item].theta)
        dataNuRef.append(pltRefAngleLogs[item].theta)

    dataAlpha = []
    dataAlphaRef = []
    for item in range(numRSA):
        dataAlpha.append(saAngleLogs[item].theta)
        dataAlphaRef.append(saRefAngleLogs[item].theta)

    np.set_printoptions(precision=16)


    # Displays the plots relative to the S/C attitude and rates errors, wheel momenta, thruster impulses, on times, and thruster firing intervals  
    
    timeData = rwMotorLog.times() * macros.NANO2HOUR

    thrLoc = thrLog.thrusterLocation
    thrDir = thrLog.thrusterDirection
    thrVec = thrLog.thrustForce_B

    plot_attitude(timeData, dataSigmaBN, dataSigmaRN, figID=1)
    plot_attitude_error(timeData, dataSigmaBR, figID=2)
    plot_rw_speeds(timeData, dataOmegaRW, numRW, figID=3)
    plot_solar_array_angle(timeData, dataAlpha, dataAlphaRef, figID=4)
    plot_platform_angle(timeData, dataNu, dataNuRef, figID=5)
    plot_thruster_cm_offset(timeData, dataRealCM, dataNu, platform.r_S1B_B, platform.dcm_S10B, thrLoc, thrDir, figID=6)
    plot_thrust_to_momentum_angle(timeData, dataOmegaRW, Gs, dataNu, platform.dcm_S10B, thrDir, figID=7)
    plot_state_errors(timeData, dataStateError, dataCovariance, figID=8)
    plot_residuals(timeData, dataPreFit, dataPostFit, cmEstimator.R0[0][0]**0.5, figID=9)

    # plot_rate_error(timeData, dataOmegaBR)
    # pltName = fileName + "2"
    # figureList[pltName] = plt.figure(2)
    #
    # plot_rw_speeds(timeData, dataOmegaRW, numRW)
    # pltName = fileName + "3"
    # figureList[pltName] = plt.figure(3)

    if show_plots:  
        plt.show()

    # close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")

    return


# Plotting RW functions
def plot_attitude(timeData, dataSigmaBN, dataSigmaRN, figID=None):
    """Plot the spacecraft attitude."""
    plt.figure(figID, figsize=(5,2.75))
    for idx in range(3):
        plt.plot(timeData, dataSigmaBN[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\sigma_{BN,' + str(idx + 1) + '}$')
    for idx in range(3):
        plt.plot(timeData, dataSigmaRN[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3), linestyle='dashed',
                 label=r'$\sigma_{RN,' + str(idx + 1) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [days]')
    plt.ylabel(r'Attitude $\sigma$')

def plot_attitude_error(timeData, dataSigmaBR, figID=None):
    """Plot the spacecraft attitude error."""
    plt.figure(figID)
    for idx in range(3):
        plt.plot(timeData, dataSigmaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\sigma_' + str(idx + 1) + '$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [days]')
    plt.ylabel(r'Attitude Tracking Error $\sigma_{B/R}$')

def plot_rw_speeds(timeData, dataOmegaRW, numRW, figID=None):
    """Plot the RW spin rates."""
    plt.figure(figID, figsize=(5,2.75))
    for idx in range(numRW):
        plt.plot(timeData, dataOmegaRW[:, idx] / macros.RPM,
                 color=unitTestSupport.getLineColor(idx, numRW),
                 label=r'$\Omega_{' + str(idx + 1) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [days]')
    plt.ylabel('RW Speed (RPM) ')

def plot_solar_array_angle(timeData, dataAngle, dataRefAngle, figID=None):
    """Plot the solar array angles."""
    plt.figure(figID)
    for i, angle in enumerate(dataAngle):
        plt.plot(timeData, angle / np.pi * 180, color='C'+str(i), label=r'$\alpha_' + str(i+1) + '$')
    for i, angle in enumerate(dataRefAngle):
        plt.plot(timeData, angle / np.pi * 180, color='C'+str(i), linestyle='dashed', label=r'$\alpha_{R,' + str(i+1) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Solar Array Angles [deg]')

def plot_platform_angle(timeData, dataAngle, dataRefAngle, figID=None):
    """Plot the solar array angles."""
    plt.figure(figID, figsize=(5,2.75))
    for i, angle in enumerate(dataAngle):
        plt.plot(timeData, angle / np.pi * 180, color='C'+str(i), label=r'$\nu_' + str(i+1) + '$')
    for i, angle in enumerate(dataRefAngle):
        plt.plot(timeData, angle / np.pi * 180, color='C'+str(i), linestyle='dashed', label=r'$\nu_{R,' + str(i+1) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [days]')
    plt.ylabel(r'Platform Angles [deg]')

def plot_thruster_cm_offset(timeData, dataCM, dataNu, dataMB_B, dataM0B, dataThrLoc_F, dataThrDir_F, figID=None):
    r_MB_B = np.array([dataMB_B[0][0], dataMB_B[1][0], dataMB_B[2][0]])
    dataAngOffset = []
    for i in range(len(timeData)):
        FM0 = [[np.cos(dataNu[1][i]),  np.sin(dataNu[0][i])*np.sin(dataNu[1][i]), -np.cos(dataNu[0][i])*np.sin(dataNu[1][i])],
               [       0      ,                   np.cos(dataNu[0][i])          ,             np.sin(dataNu[0][i])          ],
               [np.sin(dataNu[1][i]), -np.sin(dataNu[0][i])*np.cos(dataNu[1][i]),  np.cos(dataNu[0][i])*np.cos(dataNu[1][i])]]
        FB = np.matmul(FM0, dataM0B)
        BF = FB.transpose()
        r_TM_B = np.matmul(BF, dataThrLoc_F[i])
        r_CT_B = dataCM[i] - r_TM_B - r_MB_B
        thrDir_B = np.matmul(BF, dataThrDir_F[i])
        dataAngOffset.append(np.arccos(min(max(np.dot(r_CT_B, thrDir_B) / np.linalg.norm(r_CT_B), -1), 1)))
        cross = np.cross(r_CT_B, thrDir_B)
        if np.arctan2(cross[1], cross[0]) < 0:
            dataAngOffset[-1] = -dataAngOffset[-1]

    dataAngOffset = np.array(dataAngOffset) * macros.R2D
    plt.figure(figID, figsize=(5,2.75))
    plt.plot(timeData, dataAngOffset, label=r'$\Delta \theta$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [days]')
    plt.ylabel('CM Offset Ang [deg]')

def plot_thrust_to_momentum_angle(timeData, dataOmegaRW, Gs, dataNu, dataM0B, dataThrDir_F, figID=None):
    dataAngle = []
    for i in range(len(timeData)):
        FM0 = [[np.cos(dataNu[1][i]),  np.sin(dataNu[0][i])*np.sin(dataNu[1][i]), -np.cos(dataNu[0][i])*np.sin(dataNu[1][i])],
               [       0      ,                   np.cos(dataNu[0][i])          ,             np.sin(dataNu[0][i])          ],
               [np.sin(dataNu[1][i]), -np.sin(dataNu[0][i])*np.cos(dataNu[1][i]),  np.cos(dataNu[0][i])*np.cos(dataNu[1][i])]]
        FB = np.matmul(FM0, dataM0B)
        BF = FB.transpose()
        thrDir_B = np.matmul(BF, dataThrDir_F[i])

        h_B = np.array([0, 0, 0])
        for j in range(len(Gs)):
            h_B = h_B + dataOmegaRW[i][j] * Gs[j]

        h_B_norm = np.linalg.norm(h_B)
        if h_B_norm == 0:
            dataAngle.append(0.0)
        else:
            dataAngle.append(np.arccos(min(max(np.dot(h_B, thrDir_B) / np.linalg.norm(h_B), -1), 1)))

    dataAngle = np.array(dataAngle) * macros.R2D
    plt.figure(figID, figsize=(5,2.75))
    plt.plot(timeData, dataAngle, color='C3', label=r'$\Delta \phi$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [days]')
    plt.ylabel('Thr-to-Momentum Angle [deg]')

def plot_state_errors(timeData, data1, data2, figID=None):
    plt.figure(figID, figsize=(5,6))
    plt.subplot(3,1,1)
    plt.plot(timeData, data1[:, 0]*1000, color='C0', linestyle='solid', label=r'$\Delta r_1$')
    plt.plot(timeData,  3*data2[:, 0]*1000, color='C0', linestyle='dashed', label=r'$\pm 3\sigma_1$')
    plt.plot(timeData, -3*data2[:, 0]*1000, color='C0', linestyle='dashed')
    # plt.ylim(-data2[1,0]*3.5, data2[1,0]*3.5)
    plt.ylim(-data2[-1,0]*30000, data2[-1,0]*30000)
    plt.legend(loc='upper right')
    plt.ylabel('$r_{CM,1}$ [mm]')
    plt.grid()
    plt.subplot(3,1,2)
    plt.plot(timeData, data1[:, 1]*1000, color='C1', linestyle='solid', label=r'$\Delta r_2$')
    plt.plot(timeData,  3*data2[:, 1]*1000, color='C1', linestyle='dashed', label=r'$\pm 3\sigma_2$')
    plt.plot(timeData, -3*data2[:, 1]*1000, color='C1', linestyle='dashed')
    # plt.ylim(-data2[1,1]*3.5, data2[1,1]*3.5)
    plt.ylim(-data2[-1,1]*30000, data2[-1,1]*30000)
    # plt.ylim(-0.25, 0.25)
    plt.legend(loc='upper right')
    plt.ylabel('$r_{CM,2}$ [mm]')
    plt.grid()
    plt.subplot(3,1,3)
    plt.plot(timeData, data1[:, 2]*1000, color='C2', linestyle='solid', label=r'$\Delta r_3$')
    plt.plot(timeData,  3*data2[:, 2]*1000, color='C2', linestyle='dashed', label=r'$\pm 3\sigma_3$')
    plt.plot(timeData, -3*data2[:, 2]*1000, color='C2', linestyle='dashed')
    # plt.ylim(-data2[1,2]*3.5, data2[1,2]*3.5)
    plt.ylim(-data2[-1,2]*30000, data2[-1,2]*30000)
    plt.legend(loc='upper right')
    plt.ylabel('$r_{CM,3}$ [mm]')
    plt.grid()
    plt.xlabel('Time [days]')

def plot_residuals(timeData, preFit, postFit, R, figID=None):
    plt.figure(figID, figsize=(5,6))
    plt.subplot(2,1,1)
    plt.plot(timeData, preFit[:, 0]*1e3, color='C0', linestyle='solid', label=r'$\rho_1$')
    plt.plot(timeData, preFit[:, 1]*1e3, color='C1', linestyle='solid', label=r'$\rho_2$')
    plt.plot(timeData, preFit[:, 2]*1e3, color='C2', linestyle='solid', label=r'$\rho_3$')
    plt.ylabel('Pre-Fit residuals [mNm]')
    plt.legend(loc='upper right')
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(timeData, postFit[:, 0]*1e3, color='C0', linestyle='dashed', label=r'$\rho_1$')
    plt.plot(timeData, postFit[:, 1]*1e3, color='C1', linestyle='dashed', label=r'$\rho_2$')
    plt.plot(timeData, postFit[:, 2]*1e3, color='C2', linestyle='dashed', label=r'$\rho_3$')
    plt.plot([timeData[0],timeData[-1]],[3000*R,3000*R], color='C3', linestyle='dashed', label=r'$\pm 3\sigma_R$')
    plt.plot([timeData[0],timeData[-1]],[-3000*R,-3000*R], color='C3', linestyle='dashed')
    plt.legend(loc='upper right')
    plt.ylabel('Post-Fit residuals [mNm]')
    plt.grid()
    plt.xlabel('Time [days]')

if __name__ == "__main__":
    run(True)
