# Villa del Chef - Sprite Studio

**Villa del Chef - Sprite Studio** es una suite de escritorio profesional en **Python 3.11+** y **PySide6** diseñada para estudios de videojuegos 2D y pixel art. Permite catalogar el dataset dorado (`personajes al 100%`), extraer automáticamente sus animaciones, aprender la cinemática del movimiento de personajes aprobados mediante medianas robustas y modelos cinemáticos articulados, transferir ese movimiento a personajes nuevos preservando al 100% su identidad visual, y exportar spritesheets de 4 columnas x 16 filas (64 frames) con metadatos estructurados para **Unity** y motores 2D.

---

## 🌟 Características Destacadas

1. **Inmutabilidad y Protección Estricta (`SourceDatasetGuard`):**
   - El dataset original (`dataset/finished_characters/approved/personajes al 100%/`) es **estrictamente READ-ONLY**. Ningún proceso modifica, aplana ni renombra la fuente original.
   - Todos los datos derivados, anotaciones, partes segmentadas y exportaciones se escriben fuera.
2. **Esqueleto Articulado de 18 Anclajes Anatómicos (V2):**
   - Detección híbrida con jerarquía de 4 prioridades (Anotación Manual > Continuidad Temporal > V2 Anatómica > V1 Proporcional).
   - Huesos de interconexión cinemática oficial para cabeza, torso, brazos y piernas.
3. **Editor Interactivo de Articulaciones (`[ 🦴 EDIT ANCHORS ]`):**
   - Lienzo ampliado 4x con fondo damero para arrastre interactivo y edición manual de anclajes.
   - Persistencia protegida en `dataset/annotations/<char>/<variant>/<anim>/<frame:02d>.json`.
4. **Segmentación por Partes Corporales & Layer Resolver:**
   - Descompone el sprite en 6 partes canónicas (`head`, `torso`, `left_arm`, `right_arm`, `left_leg`, `right_leg`) con conservación exacta de píxeles (sin pérdidas ni duplicados).
   - Asignación dinámica de orden de profundidad (Z-Index / Painter's Algorithm) según la orientación (`down`, `up`, `left`, `right`).
5. **Plantillas Cinemáticas Articuladas V2 con Filtrado MAD:**
   - Aprendizaje de desplazamientos $(\Delta x, \Delta y)$ y rotaciones $(\Delta \theta)$ por articulación y paso de tiempo.
   - Filtrado robusto de anomalías estadísticas mediante Desviación Absoluta de la Mediana (*Median Absolute Deviation*).
6. **Transferencia Articulada V2 (`[ 🚀 GENERATE V2 ]`):**
   - **`HeadIdentityLock`:** La cabeza y rostro original se conservan 100% bit-exactos, trasladándose sin ninguna deformación ni regeneración artificial.
   - **`pixel_rotate`:** Rotación *Nearest-Neighbor* que preserva el pivote en el lienzo sin difuminados ni anti-aliasing borroso.
   - **`PaletteGuard`:** Saneamiento cromático para asegurar que ningún color ajeno a la paleta entre al sprite final.
7. **Detección Avanzada de Spritesheets V2:**
   - Detección de 16 filas x 4 columnas con soporte para fondos transparentes (alfa) o colores sólidos / chroma-key.
   - Aislamiento de figura en `tight_bbox` y partición adaptativa por valles de proyección.
   - Generación de overlay visual de depuración de cuadrícula (`debug_sheet_grid.png`).
8. **Comparador Lado a Lado V1 vs V2 (`[ ⚖️ COMPARAR V1 vs V2 ]`):**
   - Reproducción sincronizada con control de velocidad (FPS) y scrubber de frames para verificar la superioridad de estabilidad y fluidez de la versión articulada V2.
9. **Exportador Unity V2 (`[ 📦 EXPORT UNITY V2 ]`):**
   - Spritesheet empaquetado 4x16 y metadata JSON enriquecida con información del rig V2 de 18 anclajes y especificaciones para el importador de Unity.

---

## 🏛️ Arquitectura Modular del Sistema

```
villa_del_chef_sprite_studio/
├── app.py                              # Punto de entrada de la aplicación de escritorio
├── requirements.txt                    # Dependencias oficiales (PySide6, Pillow, NumPy, OpenCV, Pytest)
├── pytest.ini                          # Configuración de pytest
├── README.md                           # Documentación técnica completa
├── config/
│   └── settings.json                   # Configuración estándar (canvas 256x192, 16 filas, 4 columnas)
├── dataset/
│   ├── finished_characters/
│   │   ├── approved/
│   │   │   └── personajes al 100%/     # DATASET DORADO (READ-ONLY): Carpeta original protegida
│   │   ├── incoming/                   # Personajes entrantes para evaluación
│   │   └── rejected/                   # Personajes descartados
│   ├── annotations/                    # Anotaciones manuales de articulaciones (JSON)
│   ├── body_parts/                     # Recortes PNG de 6 partes y meta.json
│   ├── extracted_frames/               # 64 frames individuales extraídos de spritesheets
│   ├── normalized/                     # Frames normalizados (256x192 con baseline común)
│   ├── templates_v2/                   # Plantillas cinemáticas articuladas V2 (JSON)
│   ├── transferred_v2/                 # Animaciones transferidas con rig V2
│   └── indexed/
│       └── dataset_index.json          # Índice maestro incremental con hashes SHA-256
├── core/
│   ├── guard.py                        # SourceDatasetGuard y SourceDatasetWriteError
│   ├── naming.py                       # Nomenclatura oficial (rnormal, rbchef, rnchef e identidades compuestas)
│   ├── pose_analyzer_v2.py             # Detección híbrida de 18 anclajes anatómicos
│   ├── annotation_manager.py           # Gestor de persistencia de anotaciones manuales
│   ├── anchor_tracker.py               # Rastreo cinemático temporal con límite de deriva (max_drift_px)
│   ├── layer_resolver.py               # Resolución de Z-Index por punto de vista (down, up, left, right)
│   ├── body_part_segmenter.py          # Segmentador anatómico vectorizado en 6 partes canónicas
│   ├── template_extractor_v2.py        # Extractor cinemático con filtrado de anomalías MAD
│   ├── palette_guard.py                # Extracción y saneamiento de paleta de color pixel art
│   ├── motion_transfer_v2.py           # Transferencia V2 con HeadIdentityLock y pixel_rotate
│   ├── sheet_detector_v2.py            # Detección avanzada de spritesheets (alfa/chroma) y overlay
│   ├── bbox_detector.py                # Detección de bounding boxes de silueta
│   ├── baseline_detector.py            # Detección de línea base del personaje
│   ├── frame_normalizer.py             # Normalización pixel-perfect
│   ├── spritesheet_builder.py          # Generador de spritesheets 4x16 y Consistency Score
│   └── dataset_scanner.py              # Escaneo seguro de estructuras anidadas
├── models/
│   ├── skeleton.py                     # Anchor, Skeleton y conexiones SKELETON_BONES
│   ├── body_part.py                    # BodyPart (bbox, pivot, parent, z_index, mask_path)
│   ├── pose_frame.py                   # PoseFrame integral con esqueleto y baseline
│   ├── articulated_motion_template.py  # ArticulatedMotionTemplate, ArticulatedFrameTemplate, PartMotion
│   ├── frame.py                        # Frame individual
│   ├── animation.py                    # Fila de animación oficial (4 frames)
│   ├── character_variant.py            # Variante (rnormal, rbchef, rnchef)
│   └── character.py                    # Identidad única de personaje
├── services/
│   ├── dataset_service.py              # Orquestador del dataset dorado e importación segura
│   ├── training_service.py             # Entrenamiento V1 y ArticulatedTrainingWorker V2
│   ├── animation_service.py            # Extracción y generación cinemática V1 y V2
│   └── export_service.py               # Exportador Unity V1 y export_unity_package_v2
├── ui/
│   ├── theme.py                        # Tema visual oscuro pixel-art studio
│   ├── main_window.py                  # Ventana principal en 3 paneles y toolbar de pipeline
│   ├── preview_panel.py                # Visor pixel-perfect con overlay "Show Skeleton"
│   ├── timeline_panel.py               # Línea de tiempo interactiva [1..4] y selector de animación
│   ├── dataset_panel.py                # Árbol de dataset navegable
│   ├── character_panel.py              # Inspector de identidades y variantes oficiales
│   ├── anchor_editor.py                # InteractiveAnchorCanvas (4x) y AnchorEditorDialog
│   └── compare_v1_v2_dialog.py         # Comparador animado lado a lado V1 vs V2
└── tests/                              # Suite de 68 tests automatizados
    ├── test_etapa_a_v2.py              # Tests Etapa A (Esqueleto 18 anclajes)
    ├── test_etapa_b_v2.py              # Tests Etapa B (Anotación manual y AnchorTracker)
    ├── test_etapa_c_v2.py              # Tests Etapa C (Segmentación y LayerResolver)
    ├── test_etapa_d_v2.py              # Tests Etapa D (Plantillas V2 y MAD)
    ├── test_etapa_e_v2.py              # Tests Etapa E (Transferencia V2, HeadIdentityLock, PaletteGuard)
    ├── test_etapa_f_v2.py              # Tests Etapa F (SheetDetectorV2 y Overlay)
    ├── test_etapa_g_v2.py              # Tests Etapa G (Integración GUI, Comparador, Unity V2)
    ├── test_source_protection_and_structure.py # Tests de protección estricta del dataset
    ├── test_dataset_fase1.py           # Tests de escaneo y nomenclatura
    ├── test_gui_fase1.py               # Tests de UI base
    └── test_pipeline_full.py           # Tests de aceptación general
```

---

## 🏷️ Nomenclatura Oficial Obligatoria

### Claves de Variantes
- `rnormal`: Ropa Normal
- `rbchef`: Ropa Blanca Chef
- `rnchef`: Ropa Negra Chef

### Identidades Compuestas Únicas
Nombres con apellido, ciudad o apodo (**ej: `diego_vallenar`, `diego_serena`, `andres_arica`, `benja_bacaba`**) representan personas distintas y únicas. El sistema **no las fusiona**, **no las trunca** y conserva el nombre exacto de la carpeta.

---

## 🎬 Las 16 Animaciones Oficiales (4 Frames por fila = 64 frames)

| Fila | Animación | Orientación | Descripción |
|:---:|:---|:---:|:---|
| **ROW 1** | `idle_down` | `down` | Reposo frente |
| **ROW 2** | `walk_down` | `down` | Caminar frente |
| **ROW 3** | `idle_up` | `up` | Reposo espalda |
| **ROW 4** | `walk_up` | `up` | Caminar espalda |
| **ROW 5** | `idle_left` | `left` | Reposo izquierda |
| **ROW 6** | `walk_left` | `left` | Caminar izquierda |
| **ROW 7** | `idle_right` | `right` | Reposo derecha |
| **ROW 8** | `walk_right` | `right` | Caminar derecha |
| **ROW 9** | `cook_down` | `down` | Cocinar frente |
| **ROW 10** | `cook_up` | `up` | Cocinar espalda |
| **ROW 11** | `cook_left` | `left` | Cocinar izquierda |
| **ROW 12** | `cook_right` | `right` | Cocinar derecha |
| **ROW 13** | `think` | `down` | Pensar |
| **ROW 14** | `pickup` | `down` | Levantar / Cargar |
| **ROW 15** | `serve` | `down` | Servir plato |
| **ROW 16** | `celebrate` | `down` | Celebrar victoria |

---

## 🚀 Guía de Uso del Sistema V2

### 1. Iniciar la Aplicación
```powershell
python app.py
```

### 2. Entrenar el Modelo Articulado (`TRAIN ARTICULATED (V2)`)
- Presiona el botón azul **`[ 🦴 TRAIN ARTICULATED (V2) ]`**.
- El sistema escaneará los personajes aprobados, extraerá sus articulaciones con `AnchorTracker`, calculará las rotaciones y desplazamientos de cada articulación y aplicará el filtrado de anomalías **MAD** para generar las 16 plantillas limpias en `dataset/templates_v2/`.

### 3. Ajustar Anclajes Manualmente (`EDIT ANCHORS`)
- Selecciona cualquier personaje y frame en la línea de tiempo.
- Presiona **`[ 🦴 EDIT ANCHORS ]`**.
- Arrastra con el cursor cualquiera de los 18 anclajes anatómicos en el canvas ampliado 4x.
- Presiona **`💾 Guardar Anotación Manual`** para persistir tus ajustes fuera del dataset original.

### 4. Generar Animaciones V2 (`GENERATE V2`)
- Selecciona el personaje de destino.
- Presiona el botón verde **`[ 🚀 GENERATE V2 ]`**.
- El motor segmentará las 6 partes corporales, aplicará las rotaciones sin desenfoque (`pixel_rotate`), mantendrá la cabeza 100% idéntica (`HeadIdentityLock`) y verificará la paleta de colores (`PaletteGuard`).

### 5. Comparar Versiones (`COMPARAR V1 vs V2`)
- Presiona **`[ ⚖️ COMPARAR V1 vs V2 ]`**.
- Visualiza la animación en reproducción simultánea lado a lado entre la versión previa y la articulada V2 para certificar la estabilidad visual.

### 6. Exportar Paquete para Unity (`EXPORT UNITY V2`)
- Presiona **`[ 📦 EXPORT UNITY V2 ]`**.
- Se generará en `projects/export/<character>/<variant>/`:
  - `*_spritesheet_v2.png`: Spritesheet maestro 4x16 de 64 frames.
  - `*_metadata_v2.json`: Metadatos enriquecidos con la configuración del rig articulado y ajustes de importador de Unity (Point filtering, no compression, 16 PPU).

---

## 🧪 Ejecución de la Suite de Pruebas

Para ejecutar las **68 pruebas automatizadas** de extremo a extremo:
```powershell
pytest tests/ -v
```
