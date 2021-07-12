#pragma once

class PidController
{
public:
    PidController( double Kp_, double Ki_, double Kd_ )
        : Kp( Kp_ ), Ki( Ki_ ), Kd( Kd_ )
    {
        p_error = 0.;
        i_error = 0.;
        d_error = 0.;
    };

    /**
     * Update each term in the PID error variables given the current error
     * @param error The current error
     */
    void UpdateErrorTerms( double error );

    /**
     * Calculate the each term of the PID error
     * @output the total command to the actuator
     */
    double OutputToActuator( double d_limit );

	void setZero();

private:
    /**
     * PID Error terms
     */
    double p_error;
    double i_error;
    double d_error;

    /**
     * PID Gain coefficients
     */
    double Kp;
    double Ki;
    double Kd;

};

