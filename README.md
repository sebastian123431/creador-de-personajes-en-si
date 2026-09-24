# Villa del Chef - Sprite Studio

**Villa del Chef - Sprite Studio** es una suite de escritorio profesional en **Python 3.11+** y **PySide6** diseñada para estudios de videojuegos 2D. Permite importar personajes pixel art, catalogar el dataset dorado (approved), extraer automáticamente sus animaciones, aprender la cinemática del movimiento de personajes aprobados mediante medianas robustas, transferir ese movimiento a personajes nuevos preservando estrictamente su identidad visual, y exportar spritesheets de 4 columnas x 16 filas (64 frames) con metadatos estructurados para **Unity**.

---

## 🏛️ Arquitectura Modular del Sistema

```
villa_del_chef_sprite_studio/
├── app.py                      # Punto de entrada de la aplicación de escritorio
├── requirements.txt            # Dependencias oficiales (PySide6, Pillow, NumPy, OpenCV, Pytest)
├── pytest.ini                  # Configuración de pytest
├── README.md                   # Documentación técnica completa
├── config/
│   └── settings.json           # Configuración estándar (canvas 256x192, 16 filas, 4 columnas)
├── dataset/
│   ├── finished_characters/
│   │   ├── approved/           # DATASET DORADO (READ-ONLY): Fuente oficial para aprender
│   │   ├── incoming/           # Personajes entrantes para evaluación y aprobación
│   │   └── rejected/           # Personajes descartados (nunca usados para aprender)
│   ├── indexed/
│   ├── normalized/             # Frames normalizados (256x192 con baseline común)
│   ├── extracted_frames/       # 64 frames individuales extraídos de spritesheets
│   └── dataset_index.json      # Índice maestro con hashes SHA-256 (Incremental Dataset)
├── core/
│   ├── naming.py               # Nomenclatura oficial (rnormal, rbchef, rnchef e identidades compuestas)
│   ├── sheet_detector.py       # Detección de celdas por proyección alfa o cuadrícula adaptativa
│   ├── frame_extractor.py      # Extracción de 64 frames (16 filas x 4 columnas)
│   ├── bbox_detector.py        # Detección de bounding boxes y segmentación cabeza/cuerpo
│   ├── baseline_detector.py    # Detección de línea base (pies en el suelo)
│   ├── frame_normalizer.py     # Normalización pixel-perfect (Image.Resampling.NEAREST)
│   ├── alpha_analyzer.py       # Análisis de transparencia y canales alfa
│   ├── character_analyzer.py   # Extracción de proporciones, morfología y colores dominantes
│   ├── pose_analyzer.py        # Estimación de 13 anclajes articulares adaptados a pixel art
│   ├── motion_analyzer.py      # Cinemática relativa (deltas corporales, cabeza, pies, props)
│   ├── template_extractor.py   # Aprendizaje geométrico robusto mediante MEDIANAS (sin mezclar caras)
│   ├── template_library.py     # Persistencia y carga de las 16 plantillas aprendidas
│   ├── template_matcher.py     # Comparador morfológico para asociar referencias afines
│   ├── compositor.py           # Superposición de props (bowl, spoon, plate, box)
│   ├── motion_transfer.py      # Transferencia de movimiento respetando la identidad original
│   ├── similarity.py           # Comparación diferencial (cv2.absdiff) y Onion Skinning
│   ├── spritesheet_builder.py  # Generador de spritesheet 4x16, metadata Unity y Consistency Score
│   ├── dataset_scanner.py      # Escaneo de carpetas y cálculo de hashes SHA-256
│   ├── dataset_indexer.py      # Generador y actualizador incremental de dataset_index.json
│   └── validation.py           # Validador de integridad del dataset y assets
├── models/
│   ├── frame.py                # Dataclass Frame (bbox, center_x, baseline_y, alpha_area)
│   ├── animation.py            # Dataclass Animation (16 filas oficiales)
│   ├── character_variant.py    # Dataclass CharacterVariant (rnormal, rbchef, rnchef)
│   ├── character.py            # Dataclass Character (identidades compuestas)
│   ├── motion_template.py      # Dataclass MotionTemplate y TemplateFrame
│   └── dataset.py              # Dataclass DatasetIndex
├── services/
│   ├── dataset_service.py      # Orquestador del dataset dorado e importación segura
│   ├── training_service.py     # Servicio de entrenamiento en QThread con TrainingReport
│   ├── animation_service.py    # Gestión, extracción y reproducción de animaciones
│   └── export_service.py       # Exportador de paquetes de animación para Unity
├── ui/
│   ├── theme.py                # Tema visual oscuro pixel-art studio
│   ├── main_window.py          # Ventana principal en 3 paneles y timeline interactiva
│   ├── dataset_panel.py        # Panel de Aprobados, Entrantes y Rechazados
│   ├── character_panel.py      # Detalle de identidades y variantes oficiales
│   ├── preview_panel.py        # Visor pixel-perfect con zoom 1x-8x y visualizador de transparencia
│   ├── timeline_panel.py       # Línea de tiempo, selector de frames [1..4], FPS y Onion Skin
│   └── comparison_panel.py     # Diálogo de comparación con cv2.absdiff
├── assets/
│   └── props/                  # Utensilios compartidos (bowl.png, spoon.png, plate.png, box.png)
└── tests/
    ├── conftest.py             # Fixtures y generador de datos sintéticos
    ├── test_dataset_fase1.py   # Tests de escaneo, nomenclatura oficial, indexer y validación
    ├── test_gui_fase1.py       # Tests de interfaz gráfica PySide6
    └── test_pipeline_full.py   # Tests de los 64 frames, normalización, templates, Unity y aceptación
```

---

## 🏷️ Nomenclatura Oficial Obligatoria

### Claves de Variantes
- `rnormal`: Ropa Normal
- `rbchef`: Ropa Blanca Chef
- `rnchef`: Ropa Negra Chef

### Identidades Compuestas Únicas
Nombres con apellido, ciudad o apodo (**ej: `diego_vallenar` y `diego_serena`**) representan personas distintas y únicas.
El sistema **no las fusiona**, **no asume duplicidad** y genera claves canónicas unívocas:
`diego_vallenar:rnormal`, `diego_serena:rbchef`, etc.

### Formato de Archivos por Personaje
```
alex/
    alex_rnormal.png            # Referencia Master rnormal
    alex_rbchef.png             # Referencia Master rbchef
    alex_rnchef.png             # Referencia Master rnchef
    movimientos_rnormal.png     # Spritesheet de animación rnormal
    movimientos_rbchef.png      # Spritesheet de animación rbchef
    movimientos_rnchef.png      # Spritesheet de animación rnchef
```

---

## 🎬 Las 16 Filas Oficiales de Animación (4 Frames por fila = 64 frames)

| Fila | Nombre | Descripción | Props Utilizados |
|:---:|:---|:---|:---|
| **ROW 1** | `idle_down` | Reposo frente | - |
| **ROW 2** | `walk_down` | Caminar frente | - |
| **ROW 3** | `idle_up` | Reposo espalda | - |
| **ROW 4** | `walk_up` | Caminar espalda | - |
| **ROW 5** | `idle_left` | Reposo izquierda | - |
| **ROW 6** | `walk_left` | Caminar izquierda | - |
| **ROW 7** | `idle_right` | Reposo derecha | - |
| **ROW 8** | `walk_right` | Caminar derecha | - |
| **ROW 9** | `cook_down` | Cocinar frente | `bowl.png`, `spoon.png` |
| **ROW 10** | `cook_up` | Cocinar espalda | - |
| **ROW 11** | `cook_left` | Cocinar izquierda | `bowl.png`, `spoon.png` |
| **ROW 12** | `cook_right` | Cocinar derecha | `bowl.png`, `spoon.png` |
| **ROW 13** | `think` | Pensar | - |
| **ROW 14** | `pickup` | Levantar / Cargar | `box.png` |
| **ROW 15** | `serve` | Servir plato | `plate.png` |
| **ROW 16** | `celebrate` | Celebrar victoria | - |

---

## 🎯 Prioridad Absoluta: Consistencia Visual > Creatividad

- **NUNCA promediar píxeles** de rostros o identidades entre personajes.
- El sistema aprende **exclusivamente la geometría y cinemática del movimiento** (medianas de desplazamientos de cabeza, torso y extremidades).
- Al transferir movimiento a un nuevo personaje, se **reutiliza y preserva su cabeza y rostro original**, evitando regeneraciones erráticas en cada frame.

---

## 📦 Exportación para Unity

Al presionar **`[ EXPORT SPRITESHEET ]`**, se genera en `projects/export/<character_id>/<variant>/`:
1. **`spritesheet.png`**: Hoja de sprites exacta de 4 columnas x 16 filas (1024 x 3072 px o estándar configurado).
2. **`frames/`**: Secuencia individual de los 64 frames en formato PNG (`idle_down_01.png` ... `celebrate_04.png`).
3. **`metadata.json`**: Metadatos listos para importar en Unity:
```json
{
    "columns": 4,
    "rows": 16,
    "cell_width": 256,
    "cell_height": 192,
    "total_frames": 64,
    "animations": {
        "idle_down": { "row": 0, "frames": 4 },
        "walk_down": { "row": 1, "frames": 4 }
    },
    "consistency_score": {
        "overall_score": "95.2%",
        "bbox_consistency": "96.0%",
        "baseline_stability": "98.5%",
        "alpha_area_stability": "94.2%"
    }
}
```

---

## 🚀 Guía de Uso Rápido

### 1. Iniciar la Aplicación de Escritorio
```powershell
python app.py
```

### 2. Importar Personajes al Dataset Dorado
- Haz clic en **`📥 Importar personajes terminados...`** y selecciona tu carpeta (ej. `personajes al 100%/`).
- Los personajes se copian a `dataset/finished_characters/approved/` y se indexan automáticamente en `dataset/dataset_index.json` sin alterar tus archivos originales.

### 3. Entrenar Plantillas de Movimiento
- Presiona **`⚡ TRAIN DATASET`**.
- El sistema extraerá los 64 frames de cada variante, normalizará en canvas 256x192, analizará los deltas cinemáticos, descartará outliers y generará las 16 plantillas en `animations/learned/`.

### 4. Transferir Movimiento a un Personaje Nuevo
- Selecciona el personaje en el panel izquierdo (con variante que tenga imagen de referencia).
- Presiona **`✨ GENERATE MOVEMENT`**.
- El motor generará los 64 frames respetando al 100% el rostro y vestimenta del personaje.

### 5. Reproducir y Comparar
- Usa la línea de tiempo inferior: presiona **PLAY**, ajusta los FPS (1 a 24), o activa **Onion Skin**.
- Usa **`🔍 COMPARAR FRAMES (DIFF)`** para ver la resta absoluta (`cv2.absdiff`) entre dos frames.

### 6. Exportar
- Presiona **`📦 EXPORT SPRITESHEET (4x16)`** para obtener el paquete completo con el reporte de **Consistency Score**.

---

## 🧪 Ejecución de Pruebas Automatizadas

Para validar los 23 tests unitarios y de integración de extremo a extremo:
```powershell
pytest tests/ -v
```
