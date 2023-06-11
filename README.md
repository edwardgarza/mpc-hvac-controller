# mpc-hvac-controller
The goal of this is to have a more intelligent heating/cooling schedule that takes into account building parameters, occupant comfort, and predicted weather patterns.

At a high level, a traditional thermostat works as a [pid](https://en.wikipedia.org/wiki/PID_controller) controller by having a set point and changing a state variable (heat/cool) to reach that set point at each point in time. The set point will change at times with a set schedule, based on the state of people in the building, or can go to vacation mode. The goal of this is to use a [model predictive controller](https://en.wikipedia.org/wiki/Model_predictive_control) to define a better trajectory than a set schedule to further minimize energy use, cost, or even carbon intensity of the electric sources. 

<h2>Assumptions</h2>

1. Cost of energy does not depend on the amount used (i.e. constant marginal rates for each time)
2. Power is not considered a factor
3. Different sources of energy are treated the same (locally generated solar and grid usage are fungible. This is implicitly assumed in (1))

<h2>Building Modeling</h2>

Buildings will be modeled as easily as possible to begin, with increasing complexity depending on desired accuracy.

<h3>Energy Transfer</h3>
<h4>Conduction</h4>

Conduction will be one of the primary drivers in this simulation because it is both relatively easy to predict temperatures and because it is probably the main contriubtor to heat loss in colder weather. This simulation will use the following formula for heat flow

q = kA/s * $\delta$ t

where q is the power flow (J/s), k is the conductivity, A is the area, and s is the thickness of the material. R value is proportional to s/k.

<h4>Radiation</h4>

Radiation will be incorporated by using the [sol-air temperature](https://en.wikipedia.org/wiki/Sol-air_temperature) to replace the outdoor temperature

<h4>Convection</h4>

Initially convection will be ignored largely because I don't know how to model the amount of convection well for a given building as that depends on 

1. Mechanical ventillation systems
2. Building leak area
3. Outdoor temperature
4. Wind speed

I am hoping this will be relatively small and constant for areas and can be compensated for by other parameters.

<h3>Building Attributes</h3>

Buildings will have windows, walls, roofs, and floors explicitly modeled. They will be modeled as simply as possible with as few objects and may not exactly map to reality. Once again, the hope here is that the model will be able to compensate with different parameters.

<h3>Heating/Cooling Modeling</h3>
Both electric resistance and combustion fuel have efficiencies independent of external temperature. However, heat pumps and air conditioner units have variable efficiencies.

From this link

https://www.nature.com/articles/s41597-019-0199-y

We can estimate COP of an air source heat pump to first order as 6.08 - 0.09 $\delta t$

For cooling I will use the same formula for the COP - 1.

<h3>Fitting</h3>

The input parameters, $\vec{p}$, will be compared with a best fit of $\vec{p} + \vec{\epsilon}$, where $\vec{\epsilon}$ is an error vector that can be augmented to improve the fit. This will be compared with the historical data to attempt to improve the modeling on an ongoing basis and will help account for imperfect models, shade, etc.
