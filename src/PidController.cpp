#include "PidController.hpp"
#include <cstdio>


double PidController::OutputToActuator( double d_limit, bool printValue )
{
    /* optional limit on derivative term */
    if( Kd * d_error > d_limit )
        return Kp * p_error + Ki * i_error + d_limit;
    if( Kd * d_error < -d_limit )
        return Kp * p_error + Ki * i_error - d_limit;

    if( printValue )
    {
        printf( "p_error = %f, i_error = %f, d_error = %f \n", p_error, i_error, d_error );
        printf( "Kp *= %f, Ki *= %f, Kd *= %f \n", Kp * p_error, Ki * i_error, Kd * d_error );
    }

    return Kp * p_error + Ki * i_error + Kd * d_error;
}


