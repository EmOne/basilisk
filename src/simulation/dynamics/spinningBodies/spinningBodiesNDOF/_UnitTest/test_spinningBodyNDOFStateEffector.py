# ISC License
#
# Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


#
#   Unit Test Script
#   Module Name:        spinningBodiesNDOF
#   Author:             João Vaz Carneiro
#   Creation Date:      October 17, 2023
#

import inspect
import os

import numpy as np
import pytest
import numpy
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
splitPath = path.split('simulation')

from Basilisk.utilities import SimulationBaseClass, unitTestSupport, macros
from Basilisk.simulation import spacecraft, spinningBodyNDOFStateEffector, gravityEffector
from Basilisk.architecture import messaging


# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail() # need to update how the RW states are defined
# provide a unique test method name, starting with test_

# @pytest.mark.parametrize("cmdTorque1, lock1, cmdTorque2, lock2", [
#     (0.0, False, 0.0, False)
#     , (0.0, True, 0.0, False)
#     , (0.0, False, 0.0, True)
#     , (0.0, True, 0.0, True)
#     , (1.0, False, -2.0, False)
# ])
def test_spinningBody(show_plots):
    r"""
    **Validation Test Description**

    This unit test sets up a spacecraft with a single-axis rotating rigid body attached to a rigid hub. The spinning
    body's center of mass is off-center from the spinning axis and the position of the axis is arbitrary. The scenario
    includes gravity acting on both the spacecraft and the effector.

    **Description of Variables Being Tested**

    In this file we are checking the principles of conservation of energy and angular momentum. Both the orbital and
    rotational energy and angular momentum must be maintained when conservative forces like gravity are present.
    Therefore, the values of the variables

    - ``finalOrbAngMom``
    - ``finalOrbEnergy``
    - ``finalRotAngMom``
    - ``finalRotEnergy``

    against their initial values.
    """
    [testResults, testMessage] = spinningBody(show_plots)
    assert testResults < 1, testMessage


def spinningBody(show_plots):
    __tracebackhide__ = True

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages

    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraftBody"

    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    #   Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.0001)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Create two hinged rigid bodies
    spinningBodyEffector = spinningBodyNDOFStateEffector.SpinningBodyNDOFStateEffector()
    spinningBodyEffector.ModelTag = "spinningBodyEffector"

    # Define properties of spinning bodies
    spinningBody1 = spinningBodyNDOFStateEffector.SpinningBody()
    spinningBody1.mass = 100.0
    spinningBody1.ISPntSc_S = [[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]]
    spinningBody1.dcm_S0P = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    spinningBody1.r_ScS_S = [[2.0], [-0.5], [0.0]]
    spinningBody1.r_SP_P = [[-2.0], [0.5], [-1.0]]
    spinningBody1.sHat_S = [[0], [0], [1]]
    spinningBody1.thetaInit = 0.0 * macros.D2R
    spinningBody1.thetaDotInit = 2.0 * macros.D2R
    spinningBody1.k = 1.0
    spinningBodyEffector.addSpinningBody(spinningBody1)

    spinningBody2 = spinningBodyNDOFStateEffector.SpinningBody()
    spinningBody2.mass = 50.0
    spinningBody2.ISPntSc_S = [[50.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 40.0]]
    spinningBody2.dcm_S0P = [[0.0, -1.0, 0.0], [0.0, .0, -1.0], [1.0, 0.0, 0.0]]
    spinningBody2.r_ScS_S = [[1.0], [0.0], [-1.0]]
    spinningBody2.r_SP_P = [[0.5], [-1.5], [-0.5]]
    spinningBody2.sHat_S = [[0], [-1], [0]]
    spinningBody2.thetaInit = 5 * macros.D2R
    spinningBody2.thetaDotInit = -1.0 * macros.D2R
    spinningBody2.k = 2.0
    spinningBodyEffector.addSpinningBody(spinningBody2)

    spinningBody3 = spinningBodyNDOFStateEffector.SpinningBody()
    spinningBody3.mass = 20.0
    spinningBody3.ISPntSc_S = [[10.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    spinningBody3.dcm_S0P = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    spinningBody3.r_ScS_S = [[-2.0], [0.5], [1.5]]
    spinningBody3.r_SP_P = [[3.0], [0.5], [1.0]]
    spinningBody3.sHat_S = [[np.sqrt(1/2)], [np.sqrt(1/2)], [0]]
    spinningBody3.thetaInit = 1.0 * macros.D2R
    spinningBody3.thetaDotInit = -0.5 * macros.D2R
    spinningBody3.k = 4.0
    spinningBodyEffector.addSpinningBody(spinningBody3)

    spinningBody4 = spinningBodyNDOFStateEffector.SpinningBody()
    spinningBody4.mass = 10.0
    spinningBody4.ISPntSc_S = [[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
    spinningBody4.dcm_S0P = [[0.0, 1.0, 0.0], [0.0, .0, 1.0], [1.0, 0.0, 0.0]]
    spinningBody4.r_ScS_S = [[-0.4], [1.3], [-0.2]]
    spinningBody4.r_SP_P = [[0.1], [-1.9], [0.8]]
    spinningBody4.sHat_S = [[np.sqrt(1/2)], [-np.sqrt(1/2)], [0]]
    spinningBody4.thetaInit = 1 * macros.D2R
    spinningBody4.thetaDotInit = -2.0 * macros.D2R
    spinningBody4.k = 0.5
    spinningBodyEffector.addSpinningBody(spinningBody4)

    # Add spinning body to spacecraft
    scObject.addStateEffector(spinningBodyEffector)

    # Define mass properties of the rigid hub of the spacecraft
    scObject.hub.mHub = 750.0
    scObject.hub.r_BcB_B = [[0.0], [0.0], [1.0]]
    scObject.hub.IHubPntBc_B = [[900.0, 0.0, 0.0], [0.0, 800.0, 0.0], [0.0, 0.0, 600.0]]

    # Set the initial values for the states
    scObject.hub.r_CN_NInit = [[-4020338.690396649], [7490566.741852513], [5248299.211589362]]
    scObject.hub.v_CN_NInit = [[-5199.77710904224], [-3436.681645356935], [1041.576797498721]]
    scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    scObject.hub.omega_BN_BInit = [[0.1], [-0.1], [0.1]]

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, spinningBodyEffector)
    unitTestSim.AddModelToTask(unitTaskName, scObject)

    # Add Earth gravity to the simulation
    earthGravBody = gravityEffector.GravBodyData()
    earthGravBody.planetName = "earth_planet_data"
    earthGravBody.mu = 0.3986004415E+15  # meters!
    earthGravBody.isCentralBody = True
    earthGravBody.useSphericalHarmParams = False
    scObject.gravField.gravBodies = spacecraft.GravBodyVector([earthGravBody])

    # Log the spacecraft state message
    datLog = scObject.scStateOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, datLog)

    # Initialize the simulation
    unitTestSim.InitializeSimulation()

    # Add energy and momentum variables to log
    unitTestSim.AddVariableForLogging(scObject.ModelTag + ".totRotEnergy", testProcessRate, 0, 0, 'double')
    unitTestSim.AddVariableForLogging(scObject.ModelTag + ".totOrbEnergy", testProcessRate, 0, 0, 'double')
    unitTestSim.AddVariableForLogging(scObject.ModelTag + ".totOrbAngMomPntN_N", testProcessRate, 0, 2, 'double')
    unitTestSim.AddVariableForLogging(scObject.ModelTag + ".totRotAngMomPntC_N", testProcessRate, 0, 2, 'double')

    # Add states to log
    theta1Data = spinningBodyEffector.spinningBodyOutMsgs[0].recorder()
    theta2Data = spinningBodyEffector.spinningBodyOutMsgs[1].recorder()
    theta3Data = spinningBodyEffector.spinningBodyOutMsgs[2].recorder()
    theta4Data = spinningBodyEffector.spinningBodyOutMsgs[3].recorder()
    unitTestSim.AddModelToTask(unitTaskName, theta1Data)
    unitTestSim.AddModelToTask(unitTaskName, theta2Data)
    unitTestSim.AddModelToTask(unitTaskName, theta3Data)
    unitTestSim.AddModelToTask(unitTaskName, theta4Data)

    # Setup and run the simulation
    stopTime = 25000*testProcessRate
    unitTestSim.ConfigureStopTime(stopTime)
    unitTestSim.ExecuteSimulation()

    # Extract the logged variables
    orbEnergy = unitTestSim.GetLogVariableData(scObject.ModelTag + ".totOrbEnergy")
    orbAngMom_N = unitTestSim.GetLogVariableData(scObject.ModelTag + ".totOrbAngMomPntN_N")
    rotAngMom_N = unitTestSim.GetLogVariableData(scObject.ModelTag + ".totRotAngMomPntC_N")
    rotEnergy = unitTestSim.GetLogVariableData(scObject.ModelTag + ".totRotEnergy")
    theta1 = theta1Data.theta
    theta1Dot = theta1Data.thetaDot
    theta2 = theta2Data.theta
    theta2Dot = theta2Data.thetaDot
    theta3 = theta3Data.theta
    theta3Dot = theta3Data.thetaDot
    theta4 = theta4Data.theta
    theta4Dot = theta4Data.thetaDot

    # Setup the conservation quantities
    initialOrbAngMom_N = [[orbAngMom_N[0, 1], orbAngMom_N[0, 2], orbAngMom_N[0, 3]]]
    finalOrbAngMom = [orbAngMom_N[-1]]
    initialRotAngMom_N = [[rotAngMom_N[0, 1], rotAngMom_N[0, 2], rotAngMom_N[0, 3]]]
    finalRotAngMom = [rotAngMom_N[-1]]
    initialOrbEnergy = [[orbEnergy[0, 1]]]
    finalOrbEnergy = [orbEnergy[-1]]
    initialRotEnergy = [[rotEnergy[0, 1]]]
    finalRotEnergy = [rotEnergy[-1]]

    # Plotting
    plt.close("all")
    plt.figure()
    plt.clf()
    plt.plot(orbAngMom_N[:, 0] * 1e-9, (orbAngMom_N[:, 1] - orbAngMom_N[0, 1]) / orbAngMom_N[0, 1],
             orbAngMom_N[:, 0] * 1e-9, (orbAngMom_N[:, 2] - orbAngMom_N[0, 2]) / orbAngMom_N[0, 2],
             orbAngMom_N[:, 0] * 1e-9, (orbAngMom_N[:, 3] - orbAngMom_N[0, 3]) / orbAngMom_N[0, 3])
    plt.xlabel('time (s)')
    plt.ylabel('Relative Difference')
    plt.title('Orbital Angular Momentum')

    plt.figure()
    plt.clf()
    plt.plot(orbEnergy[:, 0] * 1e-9, (orbEnergy[:, 1] - orbEnergy[0, 1]) / orbEnergy[0, 1])
    plt.xlabel('time (s)')
    plt.ylabel('Relative Difference')
    plt.title('Orbital Energy')

    plt.figure()
    plt.clf()
    plt.plot(rotAngMom_N[:, 0] * 1e-9, (rotAngMom_N[:, 1] - rotAngMom_N[0, 1]) / rotAngMom_N[0, 1],
             rotAngMom_N[:, 0] * 1e-9, (rotAngMom_N[:, 2] - rotAngMom_N[0, 2]) / rotAngMom_N[0, 2],
             rotAngMom_N[:, 0] * 1e-9, (rotAngMom_N[:, 3] - rotAngMom_N[0, 3]) / rotAngMom_N[0, 3])
    plt.xlabel('time (s)')
    plt.ylabel('Relative Difference')
    plt.title('Rotational Angular Momentum')

    plt.figure()
    plt.clf()
    plt.plot(rotEnergy[:, 0] * 1e-9, (rotEnergy[:, 1] - rotEnergy[0, 1]) / rotEnergy[0, 1])
    plt.xlabel('time (s)')
    plt.ylabel('Relative Difference')
    plt.title('Rotational Energy')

    plt.figure()
    plt.clf()
    plt.plot(theta1Data.times() * 1e-9, theta1, label=r'$\theta_1$')
    plt.plot(theta2Data.times() * 1e-9, theta2, label=r'$\theta_2$')
    plt.plot(theta3Data.times() * 1e-9, theta3, label=r'$\theta_1$')
    plt.plot(theta4Data.times() * 1e-9, theta4, label=r'$\theta_2$')
    plt.legend(loc='best')
    plt.xlabel('time (s)')
    plt.ylabel('Angle')

    plt.figure()
    plt.clf()
    plt.plot(theta1Data.times() * 1e-9, theta1Dot, label=r'$\dot{\theta}_1$')
    plt.plot(theta2Data.times() * 1e-9, theta2Dot, label=r'$\dot{\theta}_2$')
    plt.plot(theta3Data.times() * 1e-9, theta3Dot, label=r'$\dot{\theta}_1$')
    plt.plot(theta4Data.times() * 1e-9, theta4Dot, label=r'$\dot{\theta}_2$')
    plt.legend(loc='best')
    plt.xlabel('time (s)')
    plt.ylabel('Angle Rate')

    if show_plots:
        plt.show()
    plt.close("all")

    # Testing setup
    accuracy = 1e-12
    finalOrbAngMom = numpy.delete(finalOrbAngMom, 0, axis=1)  # remove time column
    finalRotAngMom = numpy.delete(finalRotAngMom, 0, axis=1)  # remove time column
    finalRotEnergy = numpy.delete(finalRotEnergy, 0, axis=1)  # remove time column
    finalOrbEnergy = numpy.delete(finalOrbEnergy, 0, axis=1)  # remove time column

    for i in range(0, len(initialOrbAngMom_N)):
        # check a vector values
        if not unitTestSupport.isArrayEqualRelative(finalOrbAngMom[i], initialOrbAngMom_N[i], 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: Spinning Body integrated test failed orbital angular momentum unit test")

    for i in range(0, len(initialRotAngMom_N)):
        # check a vector values
        if not unitTestSupport.isArrayEqualRelative(finalRotAngMom[i], initialRotAngMom_N[i], 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: Spinning Body integrated test failed rotational angular momentum unit test")

    for i in range(0, len(initialRotEnergy)):
        # check a vector values
        if not unitTestSupport.isArrayEqualRelative(finalRotEnergy[i], initialRotEnergy[i], 1, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: Spinning Body integrated test failed rotational energy unit test")

    for i in range(0, len(initialOrbEnergy)):
        # check a vector values
        if not unitTestSupport.isArrayEqualRelative(finalOrbEnergy[i], initialOrbEnergy[i], 1, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: Spinning Body integrated test failed orbital energy unit test")

    if testFailCount == 0:
        print("PASSED: " + " Spinning Body gravity integrated test")

    assert testFailCount < 1, testMessages
    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]


if __name__ == "__main__":
    spinningBody(True)
