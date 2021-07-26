#pragma once

class PidController
{
public:
    PidController( double Kp_, double Ki_, double Kd_ ) : Kp( Kp_ ), Ki( Ki_ ), Kd( Kd_ )
    {
        p_error = 0.;
        i_error = 0.;
        d_error = 0.;
    }

    /**
     * Update each term in the PID error variables given the current error
     * @param error The current error
     */
    inline void UpdateErrorTerms( double error )
    {
        d_error = error - p_error;
        i_error += error;
        p_error = error;
    }

    /**
     * Calculate the each term of the PID error
     * @param printValue print the error and result in pid controller
     * @output the total command to the actuator
     */
    double OutputToActuator( double d_limit, bool printValue );

	/*set the i_error in pid controller to 0*/
    inline void setZero()
    {
        i_error = p_error = 0;
    }

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
