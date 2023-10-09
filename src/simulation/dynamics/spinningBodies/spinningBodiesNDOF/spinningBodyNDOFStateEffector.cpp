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

#include "spinningBodyNDOFStateEffector.h"
#include "architecture/utilities/avsEigenSupport.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include <string>

/*! This is the constructor, setting variables to default values */
SpinningBodyNDOFStateEffector::SpinningBodyNDOFStateEffector()
{
    // Zero the mass props and mass prop rates contributions
    this->effProps.mEff = 0.0;
    this->effProps.rEff_CB_B.fill(0.0);
    this->effProps.IEffPntB_B.fill(0.0);
    this->effProps.rEffPrime_CB_B.fill(0.0);
    this->effProps.IEffPrimePntB_B.fill(0.0);
    
    this->nameOfThetaState = "spinningBodyTheta" + std::to_string(SpinningBodyNDOFStateEffector::effectorID);
    this->nameOfThetaDotState = "spinningBodyThetaDot" + std::to_string(SpinningBodyNDOFStateEffector::effectorID);
    SpinningBodyNDOFStateEffector::effectorID++;
}

uint64_t SpinningBodyNDOFStateEffector::effectorID = 1;

/*! This is the destructor, nothing to report here */
SpinningBodyNDOFStateEffector::~SpinningBodyNDOFStateEffector()
{
    SpinningBodyNDOFStateEffector::effectorID --;    /* reset the panel ID*/
}

/*! This method is used to reset the module. */
void SpinningBodyNDOFStateEffector::Reset(uint64_t CurrentClock)
{
    for(auto& spinningBody: this->spinningBodyVec) {
        if (spinningBody.sHat_S.norm() > 0.01) {
            spinningBody.sHat_S.normalize();
        }
        else {
            bskLogger.bskLog(BSK_ERROR, "Norm of sHat must be greater than 0. sHat may not have been set by the user.");
        }
    }
}

void SpinningBodyNDOFStateEffector::addSpinningBody(const SpinningBody& newBody) {
    // Pushback new spinning body
    spinningBodyVec.push_back(newBody);
    this->N++;
    
    // Create the output vectors
    this->spinningBodyConfigLogOutMsgs.push_back(new Message<SCStatesMsgPayload>);
    this->spinningBodyOutMsgs.push_back(new Message<HingedRigidBodyMsgPayload>);

    this->ATheta.conservativeResize(this->ATheta.rows()+1, 3);
    this->BTheta.conservativeResize(this->BTheta.rows()+1, 3);
    this->CTheta.conservativeResize(this->CTheta.rows()+1);
}


/*! This method takes the computed theta states and outputs them to the messaging system. */
void SpinningBodyNDOFStateEffector::writeOutputStateMessages(uint64_t CurrentClock)
{
    // Write out the spinning body output messages
    int i = 0;
    HingedRigidBodyMsgPayload spinningBodyBuffer;
    SCStatesMsgPayload configLogMsg;
    for(auto& spinningBody: this->spinningBodyVec) {
        if (this->spinningBodyOutMsgs[i]->isLinked()) {
            spinningBodyBuffer = this->spinningBodyOutMsgs[i]->zeroMsgPayload;
            spinningBodyBuffer.theta = spinningBody.theta;
            spinningBodyBuffer.thetaDot = spinningBody.thetaDot;
            this->spinningBodyOutMsgs[i]->write(&spinningBodyBuffer, this->moduleID, CurrentClock);
        }

        if (this->spinningBodyConfigLogOutMsgs[i]->isLinked()) {
            configLogMsg = this->spinningBodyConfigLogOutMsgs[i]->zeroMsgPayload;

            // Logging the S frame is the body frame B of that object
            eigenVector3d2CArray(spinningBody.r_ScN_N, configLogMsg.r_BN_N);
            eigenVector3d2CArray(spinningBody.v_ScN_N, configLogMsg.v_BN_N);
            eigenVector3d2CArray(spinningBody.sigma_SN, configLogMsg.sigma_BN);
            eigenVector3d2CArray(spinningBody.omega_SN_S, configLogMsg.omega_BN_B);
            this->spinningBodyConfigLogOutMsgs[i]->write(&configLogMsg, this->moduleID, CurrentClock);
        }

        i++;
    }
}

/*! This method prepends the name of the spacecraft for multi-spacecraft simulations.*/
void SpinningBodyNDOFStateEffector::prependSpacecraftNameToStates()
{
    this->nameOfThetaState = this->nameOfSpacecraftAttachedTo + this->nameOfThetaState;
    this->nameOfThetaDotState = this->nameOfSpacecraftAttachedTo + this->nameOfThetaDotState;
}

/*! This method allows the SB state effector to have access to the hub states and gravity*/
void SpinningBodyNDOFStateEffector::linkInStates(DynParamManager& statesIn)
{
    this->inertialPositionProperty = statesIn.getPropertyReference(this->nameOfSpacecraftAttachedTo + "r_BN_N");
    this->inertialVelocityProperty = statesIn.getPropertyReference(this->nameOfSpacecraftAttachedTo + "v_BN_N");
}

/*! This method allows the SB state effector to register its states: theta and thetaDot with the dynamic parameter manager */
void SpinningBodyNDOFStateEffector::registerStates(DynParamManager& states)
{
    // Register the theta states
    this->thetaState = states.registerState(N, 1, this->nameOfThetaState);
    this->thetaDotState = states.registerState(N, 1, this->nameOfThetaDotState);
    Eigen::MatrixXd thetaInitMatrix(N,1);
    Eigen::MatrixXd thetaDotInitMatrix(N,1);
    int i = 0;
    for(const auto& spinningBody: this->spinningBodyVec) {
        thetaInitMatrix(i,0) = spinningBody.thetaInit;
        thetaDotInitMatrix(i,0) = spinningBody.thetaDotInit;
        i++;
    }
    this->thetaState->setState(thetaInitMatrix);
    this->thetaDotState->setState(thetaDotInitMatrix);
}

/*! This method allows the SB state effector to provide its contributions to the mass props and mass prop rates of the
 spacecraft */
void SpinningBodyNDOFStateEffector::updateEffectorMassProps(double integTime)
{
    this->effProps.mEff = 0.0;
    this->effProps.rEff_CB_B = Eigen::Vector3d::Zero();
    this->effProps.rEffPrime_CB_B = Eigen::Vector3d::Zero();
    this->effProps.IEffPntB_B = Eigen::Matrix3d::Zero();
    this->effProps.IEffPrimePntB_B = Eigen::Matrix3d::Zero();

    int i = 0;
    for(auto& spinningBody: this->spinningBodyVec) {
        // Give the mass of the spinning body to the effProps mass
        this->effProps.mEff += spinningBody.mass;

        // Grab current states
        spinningBody.theta = this->thetaState->getState()(i, 0);
        spinningBody.thetaDot = this->thetaDotState->getState()(i, 0);

        // Compute the DCM from both S frames to B frame
        double dcm_S0S[3][3];
        double prv_S0S_array[3];
        Eigen::Vector3d prv_S0S = - spinningBody.theta * spinningBody.sHat_S;
        eigenVector3d2CArray(prv_S0S, prv_S0S_array);
        PRV2C(prv_S0S_array, dcm_S0S);
        if (i == 0) {
            spinningBody.dcm_BS = spinningBody.dcm_S0P.transpose() * c2DArray2EigenMatrix3d(dcm_S0S);
        } else {
            spinningBody.dcm_BS = this->spinningBodyVec[i-1].dcm_BS * spinningBody.dcm_S0P.transpose() * c2DArray2EigenMatrix3d(dcm_S0S);
        }

        // Write the spinning axis in B frame
        spinningBody.sHat_B = spinningBody.dcm_BS * spinningBody.sHat_S;

        // Compute the effector's CoM with respect to point B
        spinningBody.r_ScS_B = spinningBody.dcm_BS * spinningBody.r_ScS_S;
        if (i == 0) {
            spinningBody.r_SP_B = spinningBody.r_SP_P;
            spinningBody.r_SB_B = spinningBody.r_SP_P;
            spinningBody.r_ScB_B = spinningBody.r_ScS_B + spinningBody.r_SB_B;
        } else {
            spinningBody.r_SP_B = this->spinningBodyVec[i-1].dcm_BS * spinningBody.r_SP_P;
            spinningBody.r_SB_B = spinningBody.r_SP_B + this->spinningBodyVec[i-1].r_SB_B;
            spinningBody.r_ScB_B = spinningBody.r_ScS_B + spinningBody.r_SB_B;
        }
        this->effProps.rEff_CB_B += spinningBody.mass * spinningBody.r_ScB_B;

        // Find the inertia of the hinged rigid bodies about point B
        spinningBody.rTilde_ScB_B = eigenTilde(spinningBody.r_ScB_B);
        spinningBody.ISPntSc_B = spinningBody.dcm_BS * spinningBody.ISPntSc_S * spinningBody.dcm_BS.transpose();
        this->effProps.IEffPntB_B += spinningBody.ISPntSc_B - spinningBody.mass * spinningBody.rTilde_ScB_B * spinningBody.rTilde_ScB_B;

        // Define omega_S1B_B, omega_S2S1_B, omega_S2B_B, and their cross product operator
        spinningBody.omega_SP_B = spinningBody.thetaDot * spinningBody.sHat_B;
        spinningBody.omegaTilde_SP_B = eigenTilde(spinningBody.omega_SP_B);
        if (i == 0) {
            spinningBody.omega_SB_B = spinningBody.omega_SP_B;
        } else {
            spinningBody.omega_SB_B = spinningBody.omega_SP_B + this->spinningBodyVec[i-1].omega_SB_B;
        }
        spinningBody.omegaTilde_SB_B = eigenTilde(spinningBody.omega_SB_B);

        // Find rPrime_Sc1B_B and rPrime_Sc2B_B
        spinningBody.rPrime_ScS_B = spinningBody.omegaTilde_SB_B * spinningBody.r_ScS_B;
        if (i == 0) {
            spinningBody.rPrime_SP_B = Eigen::Vector3d::Zero();
            spinningBody.rPrime_SB_B = spinningBody.rPrime_SP_B;
        } else {
            spinningBody.rPrime_SP_B = this->spinningBodyVec[i-1].omegaTilde_SB_B * spinningBody.r_SP_B;
            spinningBody.rPrime_SB_B = spinningBody.rPrime_SP_B + this->spinningBodyVec[i-1].rPrime_SB_B;
        }
        spinningBody.rPrime_ScB_B = spinningBody.rPrime_ScS_B + spinningBody.rPrime_SB_B;
        this->effProps.rEffPrime_CB_B += spinningBody.mass * spinningBody.rPrime_ScB_B;

        // Find the body-frame time derivative of the inertias of each spinner
        spinningBody.IPrimeSPntSc_B = spinningBody.omegaTilde_SB_B * spinningBody.ISPntSc_B - spinningBody.ISPntSc_B * spinningBody.omegaTilde_SB_B;

        // Find body time derivative of IPntSc_B
        Eigen::Matrix3d rPrimeTilde_ScB_B = eigenTilde(spinningBody.rPrime_ScB_B);
        this->effProps.IEffPrimePntB_B += spinningBody.IPrimeSPntSc_B - spinningBody.mass * (rPrimeTilde_ScB_B * spinningBody.rTilde_ScB_B + spinningBody.rTilde_ScB_B * rPrimeTilde_ScB_B);

        i++;
    }
    this->effProps.rEff_CB_B /= this->effProps.mEff;
    this->effProps.rEffPrime_CB_B /= this->effProps.mEff;
}

/*! This method allows the SB state effector to give its contributions to the matrices needed for the back-sub 
 method */
void SpinningBodyNDOFStateEffector::updateContributions(double integTime, BackSubMatrices & backSubContr, Eigen::Vector3d sigma_BN, Eigen::Vector3d omega_BN_B, Eigen::Vector3d g_N)
{
    // Find the DCM from N to B frames
    this->sigma_BN = sigma_BN;
    this->dcm_BN = (this->sigma_BN.toRotationMatrix()).transpose();

    this->omega_BN_B = omega_BN_B;
    Eigen::Matrix3d omegaTilde_BN_B = eigenTilde(this->omega_BN_B);

    // Map gravity to body frame
    Eigen::Vector3d gLocal_N;
    Eigen::Vector3d g_B;
    gLocal_N = g_N;
    g_B = this->dcm_BN * gLocal_N;

    // Compute MTheta
    Eigen::MatrixXd MTheta(this->N, this->N);
    for (int n = 0; n<this->N; n++) {
        this->spinningBodyVec[n].omega_SN_B = this->spinningBodyVec[n].omega_SB_B + this->omega_BN_B;

        for (int i = 0; i<this->N; i++) {
            MTheta(n,i) = 0.0;
            for (int j = (i<=n) ? n : i; j<this->N; j++) {
                Eigen::Vector3d r_ScjSn_B = this->spinningBodyVec[j].r_ScS_B;
                Eigen::Vector3d r_ScjSi_B = this->spinningBodyVec[j].r_ScS_B;
                for (int k = j; k>n; k--) {
                    r_ScjSn_B += this->spinningBodyVec[k].r_SP_B;
                }
                for (int k = j; k>i; k--) {
                    r_ScjSi_B += this->spinningBodyVec[k].r_SP_B;
                }
                Eigen::Matrix3d rTilde_ScjSn_B = eigenTilde(r_ScjSn_B);
                Eigen::Matrix3d rTilde_ScjSi_B = eigenTilde(r_ScjSi_B);

                MTheta(n,i) += this->spinningBodyVec[n].sHat_B.transpose() * (this->spinningBodyVec[j].ISPntSc_B
                        - this->spinningBodyVec[j].mass * rTilde_ScjSn_B * rTilde_ScjSi_B) * this->spinningBodyVec[i].sHat_B;
            }
        }
    }

    // Compute AThetaStar, BThetaStar and CThetaStar
    Eigen::MatrixXd AThetaStar(this->N, 3);
    Eigen::MatrixXd BThetaStar(this->N, 3);
    Eigen::VectorXd CThetaStar(this->N);
    for (int n = 0; n<this->N; n++) {
        AThetaStar.row(n) = Eigen::Vector3d::Zero().transpose();
        BThetaStar.row(n) = Eigen::Vector3d::Zero().transpose();
        CThetaStar(n, 0) = this->spinningBodyVec[n].u
                           - this->spinningBodyVec[n].k * this->spinningBodyVec[n].theta
                           - this->spinningBodyVec[n].c * this->spinningBodyVec[n].thetaDot;
        for (int i = n; i<this->N; i++) {
            Eigen::Vector3d r_SciSn_B = this->spinningBodyVec[i].r_ScS_B;
            for (int k = i; k>n; k--) {
                r_SciSn_B += this->spinningBodyVec[k].r_SP_B;
            }
            Eigen::Matrix3d rTilde_SciSn_B = eigenTilde(r_SciSn_B);
            Eigen::Matrix3d rTilde_SciB_B = eigenTilde(this->spinningBodyVec[i].r_ScB_B);
            Eigen::Matrix3d omegaTilde_SiN_B = eigenTilde(this->spinningBodyVec[i].omega_SN_B);

            AThetaStar.row(n) -= this->spinningBodyVec[n].sHat_B.transpose() * this->spinningBodyVec[i].mass * rTilde_SciSn_B;
            BThetaStar.row(n) -= this->spinningBodyVec[n].sHat_B.transpose() * (this->spinningBodyVec[i].ISPntSc_B
                    - this->spinningBodyVec[i].mass * rTilde_SciSn_B * rTilde_SciB_B);
            CThetaStar(n, 0) -= this->spinningBodyVec[n].sHat_B.transpose() * (
                        omegaTilde_SiN_B * this->spinningBodyVec[i].ISPntSc_B * this->spinningBodyVec[i].omega_SN_B
                        - this->spinningBodyVec[i].ISPntSc_B * this->spinningBodyVec[i].omegaTilde_SB_B * this->omega_BN_B
                        + this->spinningBodyVec[i].mass * rTilde_SciSn_B * (
                            - g_B
                            + omegaTilde_BN_B * omegaTilde_BN_B * this->spinningBodyVec[i].r_ScB_B
                            + 2 * omegaTilde_BN_B * this->spinningBodyVec[i].rPrime_ScB_B
                            + this->spinningBodyVec[i].omegaTilde_SP_B * this->spinningBodyVec[i].rPrime_ScS_B));

            for(int j=0; j<=i-1; j++) {
                Eigen::Vector3d omega_SiSj_B = Eigen::Vector3d::Zero();
                Eigen::Vector3d r_SciSj1 = this->spinningBodyVec[i].r_ScS_B;
                Eigen::Vector3d rPrime_SciSj_B = this->spinningBodyVec[i].rPrime_ScS_B;
                for (int k = i; k>j; k--) {
                    omega_SiSj_B += this->spinningBodyVec[k].omega_SP_B;
                    rPrime_SciSj_B += this->spinningBodyVec[k].rPrime_SP_B;
                }
                for (int k = i; k>j+1; k--) {
                    r_SciSj1 += this->spinningBodyVec[k].r_SP_B;
                }
                Eigen::Matrix3d omegaTilde_SiSj_B = eigenTilde(omega_SiSj_B);
                Eigen::Matrix3d rTilde_SciSj1 = eigenTilde(r_SciSj1);

                CThetaStar(n, 0) -= this->spinningBodyVec[n].sHat_B.transpose() * (
                    - this->spinningBodyVec[i].ISPntSc_B * omegaTilde_SiSj_B * this->spinningBodyVec[j].omega_SP_B
                    + this->spinningBodyVec[i].mass * rTilde_SciSn_B * (
                        this->spinningBodyVec[j].omegaTilde_SP_B * rPrime_SciSj_B
                        - rTilde_SciSj1 * this->spinningBodyVec[j].omegaTilde_SB_B * this->spinningBodyVec[j+1].omega_SP_B));
            }
        }
    }

    // Define the ATheta, BTheta and CTheta matrices
    this->ATheta = MTheta.inverse() * AThetaStar;
    this->BTheta = MTheta.inverse() * BThetaStar;
    this->CTheta = MTheta.inverse() * CThetaStar;

    // For documentation on contributions see Vaz Carneiro, Allard, Schaub spinning body paper
    for (int i = 0; i<this->N; i++) {
        Eigen::Matrix3d omegaTilde_SiN_B = eigenTilde(this->spinningBodyVec[i].omega_SN_B);
        backSubContr.vecRot -= omegaTilde_SiN_B * this->spinningBodyVec[i].ISPntSc_B * this->spinningBodyVec[i].omega_SB_B
            + this->spinningBodyVec[i].mass * omegaTilde_BN_B * this->spinningBodyVec[i].rTilde_ScB_B * this->spinningBodyVec[i].rPrime_ScB_B;

        for (int j = i; j < this->N; j++) {
            Eigen::Vector3d r_ScjSi = this->spinningBodyVec[j].r_ScS_B;
            for (int k = j; k>i; k--) {
                r_ScjSi += this->spinningBodyVec[k].r_SP_B;
            }
            Eigen::Matrix3d rTilde_ScjSi = eigenTilde(r_ScjSi);
            Eigen::Matrix3d rTilde_ScjB = eigenTilde(this->spinningBodyVec[j].r_ScB_B);

            // Translation contributions
            backSubContr.matrixA -= this->spinningBodyVec[j].mass * rTilde_ScjSi * this->spinningBodyVec[i].sHat_B * this->ATheta.row(i);
            backSubContr.matrixB -= this->spinningBodyVec[j].mass * rTilde_ScjSi * this->spinningBodyVec[i].sHat_B * this->BTheta.row(i);
            backSubContr.vecTrans += this->spinningBodyVec[j].mass * rTilde_ScjSi * this->spinningBodyVec[i].sHat_B * this->CTheta.row(i);

            // Rotation contributions
            backSubContr.matrixC += (this->spinningBodyVec[j].ISPntSc_B
                                - this->spinningBodyVec[j].mass * rTilde_ScjB * rTilde_ScjSi)
                                * this->spinningBodyVec[i].sHat_B * this->ATheta.row(i);
            backSubContr.matrixD += (this->spinningBodyVec[j].ISPntSc_B
                                - this->spinningBodyVec[j].mass * rTilde_ScjB * rTilde_ScjSi)
                                * this->spinningBodyVec[i].sHat_B * this->BTheta.row(i);
            backSubContr.vecRot -= (this->spinningBodyVec[j].ISPntSc_B
                                - this->spinningBodyVec[j].mass * rTilde_ScjB * rTilde_ScjSi)
                                * this->spinningBodyVec[i].sHat_B * this->CTheta.row(i);
        }

        for(int j=0; j<=i-1; j++) {
            Eigen::Vector3d omega_SiSj_B = Eigen::Vector3d::Zero();
            Eigen::Vector3d r_SciSj1 = this->spinningBodyVec[i].r_ScS_B;
            Eigen::Vector3d rPrime_SciSj_B = this->spinningBodyVec[i].rPrime_ScS_B;
            for (int k = i; k>j; k--) {
                omega_SiSj_B += this->spinningBodyVec[k].omega_SP_B;
                rPrime_SciSj_B += this->spinningBodyVec[k].rPrime_SP_B;
            }
            for (int k = i; k>j+1; k--) {
                r_SciSj1 += this->spinningBodyVec[k].r_SP_B;
            }
            Eigen::Matrix3d omegaTilde_SiSj_B = eigenTilde(omega_SiSj_B);
            Eigen::Matrix3d rTilde_SciSj1 = eigenTilde(r_SciSj1);

            //
            // ARE THESE SIGNS WRONG?
            //
            backSubContr.vecTrans -= this->spinningBodyVec[i].mass * (this->spinningBodyVec[j].omegaTilde_SP_B * rPrime_SciSj_B
                    - rTilde_SciSj1 * this->spinningBodyVec[j].omegaTilde_SB_B * this->spinningBodyVec[j+1].omega_SP_B);
            backSubContr.vecRot -= - this->spinningBodyVec[i].ISPntSc_B * omegaTilde_SiSj_B * this->spinningBodyVec[j].omega_SP_B
                    + this->spinningBodyVec[i].mass * this->spinningBodyVec[i].rTilde_ScB_B * (
                            this->spinningBodyVec[j].omegaTilde_SP_B * rPrime_SciSj_B
                            - rTilde_SciSj1 * this->spinningBodyVec[j].omegaTilde_SB_B * this->spinningBodyVec[j+1].omega_SP_B);
        }
        backSubContr.vecTrans -= this->spinningBodyVec[i].mass * this->spinningBodyVec[i].omegaTilde_SP_B * this->spinningBodyVec[i].rPrime_ScS_B;
        backSubContr.vecRot -= this->spinningBodyVec[i].mass * this->spinningBodyVec[i].rTilde_ScB_B * this->spinningBodyVec[i].omegaTilde_SP_B * this->spinningBodyVec[i].rPrime_ScS_B;
    }
}

/*! This method is used to find the derivatives for the SB stateEffector: thetaDDot and the kinematic derivative */
void SpinningBodyNDOFStateEffector::computeDerivatives(double integTime, Eigen::Vector3d rDDot_BN_N, Eigen::Vector3d omegaDot_BN_B, Eigen::Vector3d sigma_BN)
{   
    // Grab omegaDot_BN_B 
    Eigen::Vector3d omegaDotLocal_BN_B;
    omegaDotLocal_BN_B = omegaDot_BN_B;

    // Find rDDotLoc_BN_B
    const Eigen::Vector3d& rDDotLocal_BN_N = rDDot_BN_N;
    Eigen::Vector3d rDDotLocal_BN_B = this->dcm_BN * rDDotLocal_BN_N;

    // Compute theta and thetaDot derivatives
    Eigen::VectorXd thetaDDot = this->ATheta * rDDotLocal_BN_B + this->BTheta * omegaDotLocal_BN_B + this->CTheta;
    this->thetaState->setDerivative(this->thetaDotState->getState());
    this->thetaDotState->setDerivative(thetaDDot);
}

/*! This method is for calculating the contributions of the SB state effector to the energy and momentum of the spacecraft */
void SpinningBodyNDOFStateEffector::updateEnergyMomContributions(double integTime, Eigen::Vector3d & rotAngMomPntCContr_B, double & rotEnergyContr, Eigen::Vector3d omega_BN_B)
{
    this->omega_BN_B = omega_BN_B;
    Eigen::Matrix3d omegaTilde_BN_B = eigenTilde(this->omega_BN_B);

    rotAngMomPntCContr_B = Eigen::Vector3d::Zero();
    rotEnergyContr = 0.0;

    for(auto& spinningBody: this->spinningBodyVec) {
        // Update omega_BN_B and omega_SN_B
        spinningBody.omega_SN_B = spinningBody.omega_SB_B + this->omega_BN_B;

        // Compute rDot_ScB_B
        spinningBody.rDot_ScB_B = spinningBody.rPrime_ScB_B + omegaTilde_BN_B * spinningBody.r_ScB_B;

        // Find rotational angular momentum contribution from hub
        rotAngMomPntCContr_B += spinningBody.ISPntSc_B * spinningBody.omega_SN_B + spinningBody.mass * spinningBody.rTilde_ScB_B * spinningBody.rDot_ScB_B;

        // Find rotational energy contribution from the hub
        rotEnergyContr += 1.0 / 2.0 * spinningBody.omega_SN_B.dot(spinningBody.ISPntSc_B * spinningBody.omega_SN_B)
                        + 1.0 / 2.0 * spinningBody.mass * spinningBody.rDot_ScB_B.dot(spinningBody.rDot_ScB_B)
                        + 1.0 / 2.0 * spinningBody.k * spinningBody.theta * spinningBody.theta;
    }
}

/*! This method computes the spinning body states relative to the inertial frame */
void SpinningBodyNDOFStateEffector::computeSpinningBodyInertialStates()
{
    for(auto& spinningBody: this->spinningBodyVec) {
        // Compute the rotational properties
        Eigen::Matrix3d dcm_SN;
        dcm_SN = spinningBody.dcm_BS.transpose() * this->dcm_BN;
        spinningBody.sigma_SN = eigenMRPd2Vector3d(eigenC2MRP(dcm_SN));
        spinningBody.omega_SN_S = spinningBody.dcm_BS.transpose() * spinningBody.omega_SN_B;

        // Compute the translation properties
        spinningBody.r_ScN_N = (Eigen::Vector3d)*this->inertialPositionProperty + this->dcm_BN.transpose() * spinningBody.r_ScB_B;
        spinningBody.v_ScN_N = (Eigen::Vector3d)*this->inertialVelocityProperty + this->dcm_BN.transpose() * spinningBody.rDot_ScB_B;
    }
}

/*! This method is used so that the simulation will ask SB to update messages */
void SpinningBodyNDOFStateEffector::UpdateState(uint64_t CurrentSimNanos)
{
    //! - Read the incoming command array
    if (this->motorTorqueInMsg.isLinked() && this->motorTorqueInMsg.isWritten()) {
        ArrayMotorTorqueMsgPayload incomingCmdBuffer;
        incomingCmdBuffer = this->motorTorqueInMsg();
        int i = 0;
        for(auto& spinningBody: this->spinningBodyVec) {
            spinningBody.u = incomingCmdBuffer.motorTorque[i];
            i++;
        }
    }
    
    //! - Zero the command buffer and read the incoming command array
    if (this->motorLockInMsg.isLinked() && this->motorLockInMsg.isWritten()) {
        ArrayEffectorLockMsgPayload incomingLockBuffer;
        incomingLockBuffer = this->motorLockInMsg();
        int i = 0;
        for(auto& spinningBody: this->spinningBodyVec) {
            spinningBody.lockFlag = incomingLockBuffer.effectorLockFlag[i];
            i++;
        }
    }

    /* Compute spinning body inertial states */
    this->computeSpinningBodyInertialStates();
    
    /* Write output messages*/
    this->writeOutputStateMessages(CurrentSimNanos);
}
