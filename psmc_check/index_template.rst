=======================
PSMC temperatures check
=======================
.. role:: red

{% if proc.errors %}
Processing Errors
-----------------
.. class:: red
{% endif %}

Summary
--------         
.. class:: borderless

====================  =============================================
Date start            {{proc.datestart}}
Date stop             {{proc.datestop}}
1PDEAAT status        {%if viols.psmc%}:red:`NOT OK`{% else %}OK{% endif%} (limit = {{proc.psmc_limit|floatformat:1}} C)
{% if opt.loaddir %}
Load directory        {{opt.loaddir}}
{% endif %}
Run time              {{proc.run_time}} by {{proc.run_user}}
Run log               `<run.dat>`_
Temperatures          `<temperatures.dat>`_
States                `<states.dat>`_
====================  =============================================

{% if viols.psmc  %}
1PDEAAT Violations
-------------------
=====================  =====================  ==================
Date start             Date stop              Max temperature
=====================  =====================  ==================
{% for viol in viols.psmc %}
{{viol.datestart}}  {{viol.datestop}}  {{viol.maxtemp|floatformat:2}}
{% endfor %}
=====================  =====================  ==================
{% else %}
No 1PDEAAT Violations
{% endif %}

.. image:: {{plots.psmc.filename}}
.. image:: {{plots.pow_sim.filename}}

=======================
PSMC Model Validation
=======================

MSID quantiles
---------------

Note: PSMC quantiles are calculated using only points where 1PDEAAT > 30 degC.

.. csv-table:: 
   :header: "MSID", "1%", "5%", "16%", "50%", "84%", "95%", "99%"
   :widths: 15, 10, 10, 10, 10, 10, 10, 10

{% for plot in plots_validation %}
{% if plot.quant01 %}
   {{plot.msid}},{{plot.quant01}},{{plot.quant05}},{{plot.quant16}},{{plot.quant50}},{{plot.quant84}},{{plot.quant95}},{{plot.quant99}}
{% endif %}
{% endfor%}

{% if valid_viols %}
Validation Violations
---------------------

.. csv-table:: 
   :header: "MSID", "Quantile", "Value", "Limit"
   :widths: 15, 10, 10, 10

{% for viol in valid_viols %}
   {{viol.msid}},{{viol.quant}},{{viol.value}},{{viol.limit|floatformat:2}}
{% endfor%}

{% else %}
No Validation Violations
{% endif %}


{% for plot in plots_validation %}
{{ plot.msid }}
-----------------------

Note: PSMC residual histograms include points where 1PDEAAT > 30 degC in
blue and points where 1PDEAAT > 40 degC in red.

Red = telemetry, blue = model

.. image:: {{plot.lines}}
.. image:: {{plot.histlog}}
.. image:: {{plot.histlin}}

{% endfor %}
