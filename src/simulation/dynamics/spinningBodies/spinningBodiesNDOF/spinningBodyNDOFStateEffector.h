/*
 ISC License

 Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

 Permission to use, copy, modify, and/or distribute this software for any
 purpose with or without fee is hereby granted, provided that the above
 copyright notice and this permission notice appear in all copies.

 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

 */

#ifndef SPINNING_BODY_N_DOF_STATE_EFFECTOR_H
#define SPINNING_BODY_N_DOF_STATE_EFFECTOR_H

#include <Eigen/Dense>
#include "simulation/dynamics/_GeneralModuleFiles/stateEffector.h"
#include "simulation/dynamics/_GeneralModuleFiles/stateData.h"
#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/utilities/avsEigenMRP.h"

#include "architecture/msgPayloadDefC/ArrayMotorTorqueMsgPayload.h"
#include "architecture/msgPayloadDefC/ArrayEffectorLockMsgPayload.h"
#include "architecture/msgPayloadDefC/SCStatesMsgPayload.h"
#include "architecture/msgPayloadDefC/HingedRigidBodyMsgPayload.h"
#include "architecture/messaging/messaging.h"

#include "architecture/utilities/bskLogging.h"

struct SpinningBody {
    // Body quantities
    double mass = 0.0;
    Eigen::Matrix3d ISPntSc_S = Eigen::Matrix3d::Zero();                                   //!< [kg-m^2] Inertia of spinning body about point Sc in S frame components
    Eigen::Vector3d r_SP_P = Eigen::Vector3d::Zero();                                     //!< [m] vector pointing from body frame B origin to spinning frame S origin in B frame components
    Eigen::Vector3d r_ScS_S = Eigen::Vector3d::Zero();                                    //!< [m] vector pointing from spinning frame S origin to point Sc (center of mass of the spinner) in S frame components

    // Hinge quantities
    Eigen::Vector3d sHat_S = Eigen::Vector3d::Zero();                                     //!< -- spinning axis in S frame components.
    Eigen::Matrix3d dcm_S0P = Eigen::Matrix3d::Identity();                                    //!< -- DCM from the body frame to the S0 frame (S frame for theta=0)
    double k = 0.0;                                             //!< [N-m/rad] torsional spring constant
    double c = 0.0;                                             //!< [N-m-s/rad] rotational damping coefficient
    double thetaInit = 0.0;                                     //!< [rad] initial spinning body angle
    double thetaDotInit = 0.0;                                  //!< [rad/s] initial spinning body angle rate

    // Scalar Properties
    double theta;
    double thetaDot;
    double u;
    int lockFlag;

    // Vector quantities
    Eigen::Vector3d sHat_B;                                     //!< -- spinning axis in S frame components.
    Eigen::Vector3d r_SP_B;                                     //!< [m] vector pointing from body frame B origin to spinning frame S origin in B frame components
    Eigen::Vector3d r_SB_B;                                     //!< [m] vector pointing from body frame B origin to spinning frame S origin in B frame components
    Eigen::Vector3d r_ScS_B;                                    //!< [m] vector pointing from spinning frame S origin to point Sc (center of mass of the spinner) in S frame components
    Eigen::Vector3d r_ScB_B;                                    //!< [m] vector pointing from spinning frame S origin to point Sc (center of mass of the spinner) in S frame components
    Eigen::Vector3d rPrime_SP_B;
    Eigen::Vector3d rPrime_SB_B;
    Eigen::Vector3d rPrime_ScS_B;     //!< [m/s] body frame time derivative of r_Sc1S1_B
    Eigen::Vector3d rPrime_ScB_B;      //!< [m/s] body frame time derivative of r_Sc1B_B
    Eigen::Vector3d rDot_ScB_B;        //!< [m/s] inertial frame time derivative of r_Sc1B_B
    Eigen::Vector3d omega_SP_B;        //!< [rad/s] angular velocity of the S1 frame wrt the B frame in B frame components
    Eigen::Vector3d omega_SB_B;        //!< [rad/s] angular velocity of the S1 frame wrt the B frame in B frame components
    Eigen::Vector3d omega_SN_B;        //!< [rad/s] angular velocity of the S2 frame wrt the N frame in B frame components

    // Matrix quantities
    Eigen::Matrix3d ISPntSc_B;                                   //!< [kg-m^2] Inertia of spinning body about point Sc in S frame components
    Eigen::Matrix3d IPrimeSPntSc_B;                                   //!< [kg-m^2] Inertia of spinning body about point Sc in S frame components
    Eigen::Matrix3d dcm_BS;            //!< -- DCM from upper spinner frame to body frame
    Eigen::Matrix3d rTilde_ScB_B;      //!< [m] tilde matrix of r_Sc2B_B
    Eigen::Matrix3d omegaTilde_SP_B;   //!< [rad/s] tilde matrix of omega_S1B_B
    Eigen::Matrix3d omegaTilde_SB_B;   //!< [rad/s] tilde matrix of omega_S1B_B

    // Inertial properties
    Eigen::Vector3d r_ScN_N;           //!< [m] position vector of upper spinning body center of mass Sc2 relative to the inertial frame origin N
    Eigen::Vector3d v_ScN_N;           //!< [m/s] inertial velocity vector of Sc1 relative to inertial frame
    Eigen::Vector3d sigma_SN;          //!< -- MRP attitude of frame S2 relative to inertial frame
    Eigen::Vector3d omega_SN_S;       //!< [rad/s] inertial upper spinning body frame angular velocity vector
};

/*! @brief spinning body state effector class */
class SpinningBodyNDOFStateEffector: public StateEffector, public SysModel {
public:

    SpinningBodyNDOFStateEffector();      //!< -- Contructor
    ~SpinningBodyNDOFStateEffector() final;     //!< -- Destructor
    void Reset(uint64_t CurrentClock) final;      //!< -- Method for reset
    void writeOutputStateMessages(uint64_t CurrentClock) final;   //!< -- Method for writing the output messages
    void UpdateState(uint64_t CurrentSimNanos) final;             //!< -- Method for updating information
    void registerStates(DynParamManager& statesIn) final;         //!< -- Method for registering the SB states
    void linkInStates(DynParamManager& states) final;             //!< -- Method for getting access to other states
    void updateContributions(double integTime,
                             BackSubMatrices& backSubContr,
                             Eigen::Vector3d sigma_BN,
                             Eigen::Vector3d omega_BN_B,
                             Eigen::Vector3d g_N) final;  //!< -- Method for back-substitution contributions
    void computeDerivatives(double integTime,
                            Eigen::Vector3d rDDot_BN_N,
                            Eigen::Vector3d omegaDot_BN_B,
                            Eigen::Vector3d sigma_BN) final;                         //!< -- Method for SB to compute its derivatives
    void updateEffectorMassProps(double integTime) final;         //!< -- Method for giving the s/c the HRB mass props and prop rates
    void updateEnergyMomContributions(double integTime,
                                      Eigen::Vector3d& rotAngMomPntCContr_B,
                                      double& rotEnergyContr,
                                      Eigen::Vector3d omega_BN_B) final;       //!< -- Method for computing energy and momentum for SBs
    void prependSpacecraftNameToStates() final;                   //!< Method used for multiple spacecraft
    void computeSpinningBodyInertialStates();               //!< Method for computing the SB's states
    void addSpinningBody(SpinningBody const& newBody); //!< class method

    std::vector<Message<HingedRigidBodyMsgPayload>*> spinningBodyOutMsgs;       //!< vector of state output messages
    std::vector<Message<SCStatesMsgPayload>*> spinningBodyConfigLogOutMsgs;     //!< vector of spinning body state config log messages
    ReadFunctor<ArrayMotorTorqueMsgPayload> motorTorqueInMsg;                   //!< -- (optional) motor torque input message name
    ReadFunctor<ArrayEffectorLockMsgPayload> motorLockInMsg;                    //!< -- (optional) motor lock input message name

    std::string nameOfThetaState;                               //!< -- identifier for the theta state data container
    std::string nameOfThetaDotState;                            //!< -- identifier for the thetaDot state data container

private:
    static uint64_t effectorID;     //!< [] ID number of this effector
    int N = 0;
    std::vector<SpinningBody> spinningBodyVec;

    // Terms needed for back substitution
    Eigen::MatrixXd ATheta;     //!< -- rDDot_BN term for back substitution
    Eigen::MatrixXd BTheta;     //!< -- omegaDot_BN term for back substitution
    Eigen::VectorXd CTheta;     //!< -- scalar term for back substitution

    // Hub properties
    Eigen::Vector3d omega_BN_B;         //!< [rad/s] angular velocity of the B frame wrt the N frame in B frame components
    Eigen::MRPd sigma_BN;               //!< -- body frame attitude wrt to the N frame in MRPs
    Eigen::Matrix3d dcm_BN;             //!< -- DCM from inertial frame to body frame

    // States
    Eigen::MatrixXd* inertialPositionProperty = nullptr;    //!< [m] r_N inertial position relative to system spice zeroBase/refBase
    Eigen::MatrixXd* inertialVelocityProperty = nullptr;    //!< [m] v_N inertial velocity relative to system spice zeroBase/refBase
    StateData* thetaState = nullptr;
    StateData* thetaDotState = nullptr;
};

#endif /* SPINNING_BODY_N_DOF_STATE_EFFECTOR_H */
