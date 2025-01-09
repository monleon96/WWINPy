Getting Started
---------------

Here's a quick start guide to help you begin using WWINPy:

1. **Import the Library**
~~~~~~~~~~~~~~~~~~~~~~~~~

   .. code-block:: python

      import wwinpy

2. **Read a WWINP File and access its data**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the WWINP file.

   .. code-block:: python

      ww = wwinpy.parser.from_file('path/to/wwinp_file')

Access the data in the header, the mesh data or de weight windows values directly.

   .. code-block:: python

      ww.header.number_of_energy_bins   # Same as ww.header.ne

         [27]

   .. code-block:: python

      ww.mesh.energy_mesh[0]            # For particle_type 0

         array([1.0000e-08, 3.0000e-08, 5.0000e-08, 1.0000e-07, 2.2500e-07,
            3.2500e-07, 4.1399e-07, 8.0000e-07, 1.0000e-06, 1.1253e-06,
            1.3000e-06, 1.8554e-06, 3.0590e-06, 1.0677e-05, 2.9023e-05,
            1.0130e-04, 5.8295e-04, 3.0354e-03, 1.5034e-02, 1.1109e-01,
            4.0762e-01, 9.0718e-01, 1.4227e+00, 1.8268e+00, 3.0119e+00,
            6.3763e+00, 2.0000e+01])

   .. code-block:: python

      ww.mesh.fine_geometry_mesh

         {'x': array([-25.        ,   1.5       ,  28.        ,  54.5       ,
            81.        , 107.5       , 109.16699982, 110.83300018,
         112.5       ]),
         'y': array([-40.        , -21.25      ,  -2.5       ,  -0.83333302,
                  0.83333302,   2.5       ,  21.25      ,  40.        ]),
         'z': array([-40.        , -21.25      ,  -2.5       ,  -0.83333302,
                  0.83333302,   2.5       ,  21.25      ,  40.        ])}

   .. code-block:: python

      ww.values.ww_values[0]            # For particle_type 0

         array([[[3.74716e+11, 4.62784e+10, 2.04099e+10, ..., 1.61974e+09,
            1.59127e+09, 1.57594e+09],
            [2.97558e+11, 4.26816e+10, 1.99507e+10, ..., 1.59839e+09,
               1.57100e+09, 1.55731e+09],
            [2.51297e+11, 4.04473e+10, 1.97289e+10, ..., 1.58933e+09,
               1.56242e+09, 1.54924e+09],
            ...,
            [3.08912e+09, 1.24770e+09, 6.99321e+08, ..., 4.39785e+07,
               4.37943e+07, 4.39171e+07],
            [2.25040e+09, 9.90649e+08, 5.80925e+08, ..., 3.90302e+07,
               3.89225e+07, 3.90854e+07],
            [1.49079e+09, 7.66075e+08, 4.95798e+08, ..., 4.03358e+07,
               4.05160e+07, 4.09321e+07]]])


3. **Query Weight Windows**
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Query weight windows by specifying parameters such as particle type, time, energy, and position coordinates. You can define each parameter as a single value or a range. Exact matches are not required—WWINPy will automatically find the closest match for you.

   .. code-block:: python

      ww.query_ww(
         particle_type=0,
         energy=(1, 20),
         x=(-20, 20),
         y=0,
         z=0
      ).to_dataframe()

   Example output:

   .. raw:: html

    <div style="overflow-x: auto; max-width: 100%;">
      <table border="1" class="dataframe" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px;">
         <thead>
            <tr style="background-color: #f2f2f2; text-align: left;">
            <th style="padding: 8px;"> </th>
            <th style="padding: 8px;">particle_type</th>
            <th style="padding: 8px;">time_start</th>
            <th style="padding: 8px;">time_end</th>
            <th style="padding: 8px;">energy_start</th>
            <th style="padding: 8px;">energy_end</th>
            <th style="padding: 8px;">x_start</th>
            <th style="padding: 8px;">x_end</th>
            <th style="padding: 8px;">y_start</th>
            <th style="padding: 8px;">y_end</th>
            <th style="padding: 8px;">z_start</th>
            <th style="padding: 8px;">z_end</th>
            <th style="padding: 8px;">ww_value</th>
            </tr>
         </thead>
         <tbody>
            <tr style="background-color: #ffffff;">
            <td style="padding: 8px;">0</td>
            <td style="padding: 8px;">0</td>
            <td style="padding: 8px;">0.0</td>
            <td style="padding: 8px;">inf</td>
            <td style="padding: 8px;">0.90718</td>
            <td style="padding: 8px;">1.4227</td>
            <td style="padding: 8px;">-25.0</td>
            <td style="padding: 8px;">1.5</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">3.257030e+10</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
            <td style="padding: 8px;">1</td>
            <td style="padding: 8px;">0</td>
            <td style="padding: 8px;">0.0</td>
            <td style="padding: 8px;">inf</td>
            <td style="padding: 8px;">0.90718</td>
            <td style="padding: 8px;">1.4227</td>
            <td style="padding: 8px;">1.5</td>
            <td style="padding: 8px;">28.0</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">2.141830e+09</td>
            </tr>
            <tr style="background-color: #ffffff;">
            <td style="padding: 8px;">2</td>
            <td style="padding: 8px;">0</td>
            <td style="padding: 8px;">0.0</td>
            <td style="padding: 8px;">inf</td>
            <td style="padding: 8px;">1.4227</td>
            <td style="padding: 8px;">1.8268</td>
            <td style="padding: 8px;">-25.0</td>
            <td style="padding: 8px;">1.5</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">1.918890e+10</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
            <td style="padding: 8px;">3</td>
            <td style="padding: 8px;">0</td>
            <td style="padding: 8px;">0.0</td>
            <td style="padding: 8px;">inf</td>
            <td style="padding: 8px;">1.4227</td>
            <td style="padding: 8px;">1.8268</td>
            <td style="padding: 8px;">1.5</td>
            <td style="padding: 8px;">28.0</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">1.468830e+09</td>
            </tr>
            <tr style="background-color: #ffffff;">
            <td style="padding: 8px;">4</td>
            <td style="padding: 8px;">0</td>
            <td style="padding: 8px;">0.0</td>
            <td style="padding: 8px;">inf</td>
            <td style="padding: 8px;">1.8268</td>
            <td style="padding: 8px;">3.0119</td>
            <td style="padding: 8px;">-25.0</td>
            <td style="padding: 8px;">1.5</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">1.378580e+10</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
            <td style="padding: 8px;">5</td>
            <td style="padding: 8px;">0</td>
            <td style="padding: 8px;">0.0</td>
            <td style="padding: 8px;">inf</td>
            <td style="padding: 8px;">1.8268</td>
            <td style="padding: 8px;">3.0119</td>
            <td style="padding: 8px;">1.5</td>
            <td style="padding: 8px;">28.0</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">1.190380e+09</td>
            </tr>
            <tr style="background-color: #ffffff;">
            <td style="padding: 8px;">6</td>
            <td style="padding: 8px;">0</td>
            <td style="padding: 8px;">0.0</td>
            <td style="padding: 8px;">inf</td>
            <td style="padding: 8px;">3.0119</td>
            <td style="padding: 8px;">6.3763</td>
            <td style="padding: 8px;">-25.0</td>
            <td style="padding: 8px;">1.5</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">9.248920e+09</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
            <td style="padding: 8px;">7</td>
            <td style="padding: 8px;">0</td>
            <td style="padding: 8px;">0.0</td>
            <td style="padding: 8px;">inf</td>
            <td style="padding: 8px;">3.0119</td>
            <td style="padding: 8px;">6.3763</td>
            <td style="padding: 8px;">1.5</td>
            <td style="padding: 8px;">28.0</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">9.358320e+08</td>
            </tr>
            <tr style="background-color: #ffffff;">
            <td style="padding: 8px;">8</td>
            <td style="padding: 8px;">0</td>
            <td style="padding: 8px;">0.0</td>
            <td style="padding: 8px;">inf</td>
            <td style="padding: 8px;">6.3763</td>
            <td style="padding: 8px;">20.0</td>
            <td style="padding: 8px;">-25.0</td>
            <td style="padding: 8px;">1.5</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">5.637660e+09</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
            <td style="padding: 8px;">9</td>
            <td style="padding: 8px;">0</td>
            <td style="padding: 8px;">0.0</td>
            <td style="padding: 8px;">inf</td>
            <td style="padding: 8px;">6.3763</td>
            <td style="padding: 8px;">20.0</td>
            <td style="padding: 8px;">1.5</td>
            <td style="padding: 8px;">28.0</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">-0.833</td>
            <td style="padding: 8px;">0.833</td>
            <td style="padding: 8px;">7.149460e+08</td>
            </tr>
         </tbody>
      </table>
      </div>


4. **Optimize Weight Windows for simulation efficiency**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modify weight windows to improve simulation efficiency. The recommended workflow is to multiply, soften and apply ratio threshold to the weight windows. This can reduce the variance reduction but increase a lot the simulation efficiency.

   .. code-block:: python

      ww.multiply(2)
      ww.soften(0.6)
      ww.apply_ratio_threshold(10)

5. **Write modified Weight Windows**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write back the modified weight windows to a new file with WWINP format.

   .. code-block:: python

      ww.write_file('path/to/output_file')


