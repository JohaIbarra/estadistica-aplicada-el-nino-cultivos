# El Niño y los cultivos de Colombia — Informe ejecutivo

*Generado el 04/11/2026 a partir de 101.032 registros municipio-cultivo-año
(2007–2025), 1092 municipios y 12 cultivos.*

## La pregunta

¿Cuánto daño causa El Niño a la agricultura colombiana, y se puede anticipar qué municipios tendrán una mala
cosecha con un año de antelación?

## Lo que encontramos

1. **A escala nacional, El Niño no reduce el rendimiento de forma apreciable.** Por cada +1 °C del índice ONI, el
   rendimiento promedio cambia +0,0 % (IC 95 % [-1,1 %; +1,8 %],
   p = 0,95 por permutación). Es un resultado nulo *preciso*: descarta pérdidas nacionales
   mayores a 1,1 % por grado.

2. **El promedio esconde ganadores y perdedores.** Con intervalos que excluyen el cero:
   - Pierden: Arroz (Caribe; -2,3 % por °C); Cacao (todas las regiones; -1,7 % por °C); Caña panelera (todas las regiones; -1,5 % por °C).
   - Ganan: Café (todas las regiones; +6,2 % por °C).

   El café de montaña se beneficia de años más secos y soleados, en línea con la cosecha récord de 2015–2016.

3. **El canal más claro es la temperatura, y actúa cultivo por cultivo.** Comparando municipios dentro del mismo año,
   cada +1 °C de temperatura
   cambia el rendimiento promedio -0,5 % (IC [-1,3 %; +0,2 %]), con
   efectos negativos claros en caña panelera (-2,2 %), maíz (-2,2 %), papa (-2,1 %) y positivos en café (+4,6 %).

4. **La calidad del dato limita lo que se puede medir.** El 31 % de los rendimientos municipales de la EVA es
   idéntico al del año anterior (dato arrastrado). Los efectos estimados deben leerse como una **cota inferior**.

## Alerta temprana 2027

El ONI de julio a septiembre de 2026 promedió **2,13 °C**: un El Niño fuerte en desarrollo.
El modelo de alerta, emitido con la información disponible al 1 de octubre de 2026, estima
**229 mil hectáreas** con una caída de rendimiento de 20 % o más en 2027 (valor esperado). Mayor
exposición: Palma de aceite (35 mil ha), Plátano (33 mil ha), Maíz (31 mil ha).

**¿Cuánto de eso es El Niño?** Con las mismas series y un ENSO neutral, el modelo estimaría 219 mil
ha: El Niño suma unas **10 mil ha** (+4,7 %). El riesgo que más sube es
el de palma de aceite (1,9 puntos porcentuales) y el que más baja, el de
fríjol (-3,6 pp). Coherente con la inferencia: El Niño redistribuye el
riesgo más que multiplicarlo. *Advertencia:* el ONI actual supera el máximo visto en entrenamiento
(1,73 °C) y un modelo de árboles no extrapola; el efecto de un evento tan intenso
podría estar subestimado.

**¿Es confiable?** En la validación histórica (pronósticos simulados 2017–2025
sin ver el futuro), el 10 % de casos marcados como de mayor riesgo tuvo una tasa real de malas cosechas de
52 %, frente al 16 % promedio: **3,3 veces más**.
ROC-AUC = 0,71; supera con holgura a la referencia de persistencia, y con etiquetas permutadas cae
al azar (ROC-AUC 0,49), lo que descarta fugas de información. Sirve
para **priorizar** visitas técnicas, crédito y seguros, no como predicción exacta.

## Recomendaciones

- **Focalizar** la respuesta a El Niño por cultivo y región, no con medidas nacionales uniformes.
- **Priorizar** el monitoreo en los municipios del decil de mayor riesgo del tablero de alerta.
- **Mejorar la EVA**: marcar explícitamente los valores no actualizados permitiría medir el impacto climático con
  mucha más precisión.

*Metodología completa, supuestos y limitaciones en `reports/informe_tecnico.md`.*
