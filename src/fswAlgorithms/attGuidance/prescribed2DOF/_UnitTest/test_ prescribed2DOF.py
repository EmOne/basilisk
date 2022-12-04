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


#
#   Unit Test Script
#   Module Name:        prescribed2DOF
#   Author:             Leah Kiner
#   Creation Date:      Nov 27, 2022
#

import pytest
import os, inspect
import numpy as np

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
bskName = 'Basilisk'
splitPath = path.split(bskName)


# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport                  # general support file with common unit test functions
from Basilisk.utilities import RigidBodyKinematics as rbk
import matplotlib.pyplot as plt
from Basilisk.fswAlgorithms import prescribed2DOF                # import the module that is to be tested
from Basilisk.utilities import macros
from Basilisk.architecture import messaging                      # import the message definitions
from Basilisk.architecture import bskLogging

ang = np.linspace(0, np.pi, 5)  # [rad]
ang = list(ang)
maxAngAccel = np.linspace(0.008, 0.015, 3)  # [rad/s^2]
maxAngAccel = list(maxAngAccel)

@pytest.mark.parametrize("thetaInit", ang)
@pytest.mark.parametrize("thetaRef1", ang)
@pytest.mark.parametrize("thetaRef2", ang)
@pytest.mark.parametrize("thetaDDotMax", maxAngAccel)
@pytest.mark.parametrize("accuracy", [1e-12])

# update "module" in this function name to reflect the module name
def test_Prescribed2DOFTestFunction(show_plots, thetaInit, thetaRef1, thetaRef2, thetaDDotMax, accuracy):

    # each test method requires a single assert method to be called
    [testResults, testMessage] = Prescribed2DOFTestFunction(show_plots, thetaInit, thetaRef1, thetaRef2, thetaDDotMax, accuracy)

    assert testResults < 1, testMessage


def Prescribed2DOFTestFunction(show_plots, thetaInit, thetaRef1, thetaRef2, thetaDDotMax, accuracy):

    testFailCount = 0                                        # zero unit test result counter
    testMessages = []                                        # create empty array to store test log messages
    unitTaskName = "unitTask"                                # arbitrary name (don't change)
    unitProcessName = "TestProcess"                          # arbitrary name (don't change)
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)     # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    Prescribed2DOFConfig = prescribed2DOF.Prescribed2DOFConfig()
    Prescribed2DOFWrap = unitTestSim.setModelDataWrap(Prescribed2DOFConfig)
    Prescribed2DOFWrap.ModelTag = "Prescribed2DOF"                                 # update python name of test module

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, Prescribed2DOFWrap, Prescribed2DOFConfig)

    # Initialize the test module configuration data
    Prescribed2DOFConfig.rotAxis1_M = np.array([1.0, 0.0, 0.0])
    Prescribed2DOFConfig.rotAxis2_M = np.array([1.0, 0.0, 0.0])
    Prescribed2DOFConfig.thetaDDotMax = thetaDDotMax  # [rad/s^2]
    Prescribed2DOFConfig.r_FM_M = np.array([1.0, 0.0, 0.0])
    Prescribed2DOFConfig.rPrime_FM_M = np.array([0.0, 0.0, 0.0])
    Prescribed2DOFConfig.rPrimePrime_FM_M = np.array([0.0, 0.0, 0.0])
    Prescribed2DOFConfig.omega_FM_F = np.array([0.0, 0.0, 0.0])
    Prescribed2DOFConfig.omegaPrime_FM_F = np.array([0.0, 0.0, 0.0])
    Prescribed2DOFConfig.sigma_FM = np.array([0.0, 0.0, 0.0])

    # Create input message
    thetaDotRef = 0.0  # [rad/s]
    thetaDotInit = 0.0  # [rad/s^2]

    SpinningBodyTwoDOFMessageData = messaging.SpinningBodyTwoDOFMsgPayload()
    SpinningBodyTwoDOFMessageData.theta1 = thetaRef1
    SpinningBodyTwoDOFMessageData.theta2 = thetaRef2
    SpinningBodyTwoDOFMessageData.thetaDot1 = thetaDotRef
    SpinningBodyTwoDOFMessageData.thetaDot2 = thetaDotRef
    SpinningBodyTwoDOFMessage = messaging.SpinningBodyTwoDOFMsg().write(SpinningBodyTwoDOFMessageData)
    Prescribed2DOFConfig.spinningBodyTwoDOFInMsg.subscribeTo(SpinningBodyTwoDOFMessage)

    # Connect module output message to its own input message
    Prescribed2DOFConfig.prescribedMotionInMsg.subscribeTo(Prescribed2DOFConfig.prescribedMotionOutMsg)

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = Prescribed2DOFConfig.prescribedMotionOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Add energy and momentum variables to log
    unitTestSim.AddVariableForLogging(Prescribed2DOFWrap.ModelTag + ".phi", testProcessRate, 0, 0, 'double')
    unitTestSim.AddVariableForLogging(Prescribed2DOFWrap.ModelTag + ".thetaAccum", testProcessRate, 0, 0, 'double')

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time
    simTime = np.sqrt(((0.5 * np.abs((thetaRef1 + thetaRef2) - thetaInit)) * 8) / thetaDDotMax)
    unitTestSim.ConfigureStopTime(macros.sec2nano(simTime))        # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    SpinningBodyTwoDOFMessageData = messaging.SpinningBodyTwoDOFMsgPayload()

    thetaRef1b = thetaRef1 + 0.5 * thetaRef1
    thetaRef2b = thetaRef2 + 0.5 * thetaRef2

    #thetaRef1b = 0
    #thetaRef2b = thetaRef2

    #thetaRef1b = 0
    #thetaRef2b = -thetaRef2

    SpinningBodyTwoDOFMessageData.theta1 = thetaRef1b
    SpinningBodyTwoDOFMessageData.theta2 = thetaRef2b
    SpinningBodyTwoDOFMessageData.thetaDot1 = thetaDotRef
    SpinningBodyTwoDOFMessageData.thetaDot2 = thetaDotRef
    SpinningBodyTwoDOFMessage = messaging.SpinningBodyTwoDOFMsg().write(SpinningBodyTwoDOFMessageData, macros.sec2nano(simTime))
    Prescribed2DOFConfig.spinningBodyTwoDOFInMsg.subscribeTo(SpinningBodyTwoDOFMessage)
    unitTestSim.ConfigureStopTime(3 * macros.sec2nano(simTime))  # seconds to stop simulation
    #breakpoint()

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    # Extract the logged variables
    phi = unitTestSim.GetLogVariableData(Prescribed2DOFWrap.ModelTag + ".phi")
    thetaAccum = unitTestSim.GetLogVariableData(Prescribed2DOFWrap.ModelTag + ".thetaAccum")
    phi = np.delete(phi, 0, axis=1)
    thetaAccum = np.delete(thetaAccum, 0, axis=1)

    # Extract logged data
    omega_FM_F = dataLog.omega_FM_F
    sigma_FM = dataLog.sigma_FM
    timespan = dataLog.times()

    thetaDot_Final = np.linalg.norm(omega_FM_F[-1, :])
    sigma_FM_Final = sigma_FM[-1, :]

    theta_FM_Final = np.linalg.norm(rbk.MRP2PRV(sigma_FM_Final))

    # Calculate required reference angle

    # Plot omega_FB_F
    plt.figure()
    plt.clf()
    plt.plot(timespan * 1e-9, omega_FM_F[:, 0],
             timespan * 1e-9, omega_FM_F[:, 1],
             timespan * 1e-9, omega_FM_F[:, 2])
    plt.xlabel('Time (s)')
    plt.ylabel('omega_FB_F (rad/s)')

    # Plot simulated theta_FM
    thetaRef1_plotting = np.ones(len(timespan)) * (thetaRef1 + thetaRef2)
    thetaRef2_plotting = np.ones(len(timespan)) * (thetaRef1b + thetaRef2b)
    thetaInit_plotting = np.ones(len(timespan)) * thetaInit
    plt.figure()
    plt.clf()
    plt.plot(timespan * 1e-9, phi, label='phi')
    plt.plot(timespan * 1e-9, thetaRef1_plotting, '--', label='thetaRef1')
    plt.plot(timespan * 1e-9, thetaRef2_plotting, '--', label='thetaRef2')
    plt.plot(timespan * 1e-9, thetaInit_plotting, '--', label='thetaInit')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta_FM (rad)')
    plt.legend()

    # Plot accumulated theta
    plt.figure()
    plt.clf()
    plt.plot(timespan * 1e-9, thetaAccum)
    plt.xlabel('Time (s)')
    plt.ylabel('Accumulated Theta (rad/s)')

    if show_plots:
        plt.show()
    plt.close("all")

    # set the filtered output truth states
    if not unitTestSupport.isDoubleEqual(thetaDot_Final, thetaDotRef, accuracy):
        testFailCount += 1
        testMessages.append("FAILED: " + Prescribed2DOFWrap.ModelTag + "thetaDot_Final and thetaDotRef do not match")

    if not unitTestSupport.isDoubleEqual(theta_FM_Final, thetaRef1 + thetaRef2, accuracy):
        testFailCount += 1
        testMessages.append("FAILED: " + Prescribed2DOFWrap.ModelTag + "theta_FM_Final and thetaRef do not match")

    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    Prescribed2DOFTestFunction(
                 True,
                 0.0,           # thetaInit
                 np.pi / 4,     # thetaRef1
                 np.pi / 4,     # thetaRef2
                 0.008,         # thetaDDotMax
                 1e-12          # accuracy
               )
