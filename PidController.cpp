#include "PidController.hpp"
#include <cstdio>

void PidController::UpdateErrorTerms( double error )
{
    d_error = error - p_error;
    i_error += error;
    p_error = error;
}

double PidController::OutputToActuator( double d_limit )
{
    /* optional limit on derivative term */
    if( Kd * d_error > d_limit )
        return Kp * p_error + Ki * i_error + d_limit;
    if( Kd * d_error < -d_limit )
        return Kp * p_error + Ki * i_error - d_limit;
    printf( "p_error = %f, i_error = %f, d_error = %f \n", p_error, i_error, d_error );
    printf( "Kp *= %f, Ki *= %f, Kd *= %f \n", Kp * p_error, Ki * i_error, Kd * d_error );
    return Kp * p_error + Ki * i_error + Kd * d_error;
}

void PidController::setZero()
{
     i_error = p_error = 0;
}