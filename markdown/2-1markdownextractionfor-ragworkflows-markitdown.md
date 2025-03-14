# 使用 MarkItDown 的 Python API：全面转换示例

MarkItDown 是一个功能强大的 Python 库，旨在将多种文件格式转换为 Markdown，便于完成文档编制、索引和文本分析等任务。在本文中，我们将探讨使用 MarkItDown 的 Python API 将各种文件类型转换为 Markdown 的实用示例。我们将涵盖位于 `/kaggle/input/unstructured-files/` 目录下的以下输入文件：

- **文档**：
  - `/kaggle/input/unstructured-files/data.pdf`
  - `/kaggle/input/unstructured-files/data.docx`
  - `/kaggle/input/unstructured-files/data.xlsx`
  - `/kaggle/input/unstructured-files/data.pptx`

- **网页和文本内容**：
  - `/kaggle/input/unstructured-files/data.html`
  - `/kaggle/input/unstructured-files/data.txt`
  - `/kaggle/input/unstructured-files/data.csv`

- **媒体文件**：
  - `/kaggle/input/unstructured-files/data.mp3`
  - `/kaggle/input/unstructured-files/bar.jpg`
  - `/kaggle/input/unstructured-files/tabular.jpg`
  - `/kaggle/input/unstructured-files/bubble.png`
  - `/kaggle/input/unstructured-files/line.png`

让我们深入每个示例，了解如何有效利用 MarkItDown 进行多样化的文件转换。

## 先决条件

在开始之前，请确保具备以下条件：

- **Python 环境**：确保已安装 Python 3.6 或更高版本。
- **MarkItDown 和 OpenAI 库**：使用 `pip` 安装 MarkItDown 和 OpenAI 库：

    ```bash
    !pip install -qU openai
    !pip install -qU markitdown
    ```

- **可选依赖**：
  
  - **大语言模型 (LLM) 集成**：如果打算使用大语言模型（如 OpenAI 的 GPT-4）来增强图像描述，请确保已安装并配置好必要的 API 客户端。
  - **音频转录**：对于转录音频文件，安装 `pydub` 和 `speech_recognition`：
  
    ```bash
    pip install pydub speechrecognition
    ```
  
    另外，确保系统中已安装 `ffmpeg`，因为 `pydub` 依赖它进行音频处理。

```python
!pip install -qU openai
!pip install -qU markitdown
```

## 初始化

首先，我们将使用 OpenAI 的 GPT-4 或 Anthropic 的 Claude-3 初始化语言模型 (LLM)。API 密钥通过 `kaggle_secrets` 安全获取。然后，导入必要的类并初始化 `MarkItDown` 转换器。如果计划使用 LLM 集成进行图像描述，请相应配置 `llm_client` 和 `llm_model`。

**说明**：

1. **安全获取 API 密钥**：
   - **`kaggle_secrets`**：利用 Kaggle 的 `UserSecretsClient` 安全获取 OpenAI API 密钥，确保脚本中不硬编码或暴露敏感信息。
   
2. **OpenAI 客户端初始化**：
   - **`OpenAI`**：使用获取的 API 密钥初始化 OpenAI 客户端，支持与 GPT-4 等模型集成，用于增强功能，如图像描述。
   
3. **MarkItDown 初始化**：
   - **`MarkItDown`**：使用 LLM 客户端初始化 MarkItDown 转换器，并指定使用的模型（本例中为 `gpt-4o`）。

```python
from openai import OpenAI
from markitdown import MarkItDown
from kaggle_secrets import UserSecretsClient

# 安全获取 API 密钥
user_secrets = UserSecretsClient()

# 初始化 OpenAI 客户端
client = OpenAI(api_key=user_secrets.get_secret("my-openai-api-key"))

# 使用 LLM 集成初始化 MarkItDown
md_converter = MarkItDown(llm_client=client, llm_model="gpt-4o")
```

---

## 转换单个文件
以下是使用 MarkItDown 的 Python API 将每种指定文件类型转换为 Markdown 的示例。

### PDF 转换
MarkItDown 可让您轻松将 PDF 文档转换为 Markdown 格式，提取文本内容，以便轻松集成到文档或分析工作流程中。

```python
# PDF 文件路径
pdf_path = "/kaggle/input/unstructured-files/data.pdf"

# 将 PDF 转换为 Markdown
pdf_result = md_converter.convert(pdf_path)

# 将 Markdown 内容保存到文件
with open("/kaggle/working/data.pdf.md", "w", encoding="utf-8") as f:
    f.write(pdf_result.text_content)

print("PDF 转换完成。输出已保存到 data.pdf.md")
```

### Word 文档转换

无缝地将 Word 文档（`.docx`）转换为 Markdown，保留标题和表格等关键格式元素，以确保文档的一致性。

**说明**：
- 内部使用 `mammoth` 将 `.docx` 转换为 HTML，然后再转换为 Markdown。
- 转换后的内容保留标题和表格等格式元素。

```python
# DOCX 文件路径
docx_path = "/kaggle/input/unstructured-files/data.docx"

# 将 DOCX 转换为 Markdown
docx_result = md_converter.convert(docx_path)

# 将 Markdown 内容保存到文件
with open("/kaggle/working/data.docx.md", "w", encoding="utf-8") as f:
    f.write(docx_result.text_content)

print("Word 文档转换完成。输出已保存到 data.docx.md")
```

### Excel 电子表格转换

高效地将 Excel 电子表格（`.xlsx`）转换为 Markdown 表格，便于在支持 Markdown 的环境中进行数据分析和报告。

**说明**：
- Excel 文件中的每个工作表都转换为单独的 Markdown 表格。
- 工作表名称用作 Markdown 文档中的章节标题。

```python
# XLSX 文件路径
xlsx_path = "/kaggle/input/unstructured-files/data.xlsx"

# 将 XLSX 转换为 Markdown
xlsx_result = md_converter.convert(xlsx_path)

# 将 Markdown 内容保存到文件
with open("/kaggle/working/data.xlsx.md", "w", encoding="utf-8") as f:
    f.write(xlsx_result.text_content)

print("Excel 电子表格转换完成。输出已保存到 data.xlsx.md")
```

### PowerPoint 演示文稿转换

将 PowerPoint 演示文稿（`.pptx`）转换为 Markdown，提取幻灯片标题、文本、图像、表格和图表，以生成全面的文档。

**说明**：
- 处理幻灯片以提取标题、文本、图像、表格和图表。
- 每张幻灯片都标注有幻灯片编号和备注（如果有）。

```python
# PPTX 文件路径
pptx_path = "/kaggle/input/unstructured-files/data.pptx"

# 将 PPTX 转换为 Markdown
pptx_result = md_converter.convert(pptx_path)

# 将 Markdown 内容保存到文件
with open("/kaggle/working/data.pptx.md", "w", encoding="utf-8") as f:
    f.write(pptx_result.text_content)

print("PowerPoint 演示文稿转换完成。输出已保存到 data.pptx.md")
```

### HTML 页面转换

将 HTML 页面（`.html`）转换为 Markdown，保留标题、链接和图像等关键元素，以简化文档编制。

**说明**：
- 解析 HTML 内容，移除脚本和样式。
- 将主要内容转换为 Markdown，保留标题、链接和图像。

```python
# HTML 文件路径
html_path = "/kaggle/input/unstructured-files/data.html"

# 将 HTML 转换为 Markdown
html_result = md_converter.convert(html_path)

# 将 Markdown 内容保存到文件
with open("/kaggle/working/data.html.md", "w", encoding="utf-8") as f:
    f.write(html_result.text_content)

print("HTML 页面转换完成。输出已保存到 data.html.md")
```

### 纯文本转换

直接将纯文本文件（`.txt`）转换为 Markdown，便于将原始文本集成到支持 Markdown 的平台，且格式调整最少。

**说明**：
- 直接从纯文本文件中提取文本内容并转换为 Markdown。
- 由于源文件为纯文本，应用最小的格式调整。

```python
# TXT 文件路径
txt_path = "/kaggle/input/unstructured-files/data.txt"

# 将 TXT 转换为 Markdown
txt_result = md_converter.convert(txt_path)

# 将 Markdown 内容保存到文件
with open("/kaggle/working/data.txt.md", "w", encoding="utf-8") as f:
    f.write(txt_result.text_content)

print("纯文本转换完成。输出已保存到 data.txt.md")
```

### CSV 转换

将 CSV 等结构化数据格式转换为 Markdown，便于在 Markdown 文档中轻松嵌入数据表格和格式化内容。

```python
# CSV 文件路径
csv_path = "/kaggle/input/unstructured-files/data.csv"

# 将 CSV 转换为 Markdown
csv_result = md_converter.convert(csv_path)

# 将 Markdown 内容保存到文件
with open("/kaggle/working/data.csv.md", "w", encoding="utf-8") as f:
    f.write(csv_result.text_content)

print("CSV 转换完成。输出已保存到 data.csv.md")
```

### MP3 音频转换

利用 MarkItDown 将 MP3 音频文件转换为 Markdown，提取元数据并转录语音，以生成全面的文档。

**说明**：
- **元数据提取**：使用 `exiftool` 提取标题、艺术家、专辑、流派和时长等元数据。
- **语音转录**：如果可用，使用 `speech_recognition` 从音频文件中转录语音。
- **输出**：在 Markdown 文件中包含元数据和转录内容。

```python
# MP3 文件路径
mp3_path = "/kaggle/input/unstructured-files/data.mp3"

# 将 MP3 转换为 Markdown
mp3_result = md_converter.convert(mp3_path)

# 将 Markdown 内容保存到文件
with open("/kaggle/working/data.mp3.md", "w", encoding="utf-8") as f:
    f.write(mp3_result.text_content)

print("MP3 音频转换完成。输出已保存到 data.mp3.md")
```

### 图像转换

将图像文件（`.jpg`、`.jpeg`、`.png`）转换为 Markdown，提取元数据并使用大语言模型生成详细描述，以增强文档效果。

**输入文件**：
- `/kaggle/input/unstructured-files/bar.jpg`
- `/kaggle/input/unstructured-files/tabular.jpg`
- `/kaggle/input/unstructured-files/bubble.png`
- `/kaggle/input/unstructured-files/line.png`

**说明**：
- **元数据提取**：使用 `exiftool` 提取图像尺寸、标题、描述等元数据。
- **基于 LLM 的描述**：使用大语言模型生成图像的详细描述。
- **输出**：在 Markdown 文件中包含元数据和生成的描述。

**对其他图像文件（`tabular.jpg`、`bubble.png`、`line.png`）重复上述步骤，只需相应更新 `image_path` 和输出文件名即可。**

```python
# 图像文件路径
image_path = "/kaggle/input/unstructured-files/bar.jpg"

# 使用基于 LLM 的描述将图像转换为 Markdown
image_result = md_converter.convert(image_path)

# 将 Markdown 内容保存到文件
with open("/kaggle/working/bar.jpg.md", "w", encoding="utf-8") as f:
    f.write(image_result.text_content)

print("图像转换完成。输出已保存到 bar.jpg.md")
```

---

## 批量处理多个文件

MarkItDown 不仅擅长转换单个文件，还能无缝处理批量转换。以下是如何使用 Python 脚本一次性转换所有指定文件的示例。

### 批量转换脚本

该脚本是一个批量文件转换工具，旨在将各种文件格式（PDF、DOCX、Excel、图像等）转换为 Markdown 格式。它使用 `MarkItDown` 库，并通过 OpenAI 的 API 提供可选的 LLM 集成，以增强转换能力。

主要功能：
- 通过 Kaggle Secrets 安全处理 API 密钥
- 全面的日志记录
- 处理不支持的格式和转换失败的错误
- 支持多种输入文件类型
- LLM 集成以改善内容解释

```python
from markitdown import MarkItDown, FileConversionException, UnsupportedFormatException
from openai import OpenAI  # 可选：用于 LLM 集成
from kaggle_secrets import UserSecretsClient
import os
import logging

# 配置日志
logging.basicConfig(
    filename='/kaggle/working/markitdown_conversion.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 安全获取 API 密钥
user_secrets = UserSecretsClient()

# 初始化 OpenAI 客户端（将 'my-openai-api-key' 替换为您的实际密钥名称）
client = OpenAI(api_key=user_secrets.get_secret("my-openai-api-key"))

# 使用 LLM 集成初始化 MarkItDown
md_converter = MarkItDown(
    llm_client=client,
    llm_model="gpt-4o"  # 指定所需的 LLM 模型
)

# 输入文件所在的目录
input_dir = "/kaggle/input/unstructured-files/"

# 输出目录
output_dir = "/kaggle/working/"

# 输入文件列表
input_files = [
    "data.pdf",
    "data.docx",
    "data.xlsx",
    "data.pptx",
    "data.html",
    "data.txt",
    "data.csv",
    "data.mp3",
    "bar.jpg",
    "tabular.jpg",
    "bubble.png",
    "line.png"
]

for file_name in input_files:
    input_path = os.path.join(input_dir, file_name)
    output_file_name = f"{file_name}.md"
    output_path = os.path.join(output_dir, output_file_name)
    
    try:
        # 检查输入文件是否存在
        if not os.path.isfile(input_path):
            logging.warning(f"未找到文件：{input_path}")
            print(f"警告：未找到文件：{file_name}")
            continue
        
        # 将文件转换为 Markdown
        result = md_converter.convert(input_path)
        
        # 将结果保存到 Markdown 文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.text_content)
        
        logging.info(f"成功将 {file_name} 转换为 {output_file_name}")
        print(f"成功将 {file_name} 转换为 {output_file_name}")
        
    except UnsupportedFormatException as ufe:
        logging.error(f"文件 {file_name} 的格式不受支持：{ufe}")
        print(f"错误：文件 {file_name} 的格式不受支持：{ufe}")
        
    except FileConversionException as fce:
        logging.error(f"文件 {file_name} 转换失败：{fce}")
        print(f"错误：文件 {file_name} 转换失败：{fce}")
        
    except Exception as e:
        logging.error(f"文件 {file_name} 出现意外错误：{e}")
        print(f"错误：文件 {file_name} 出现意外错误：{e}")

print("\n批量转换完成。请查看 'markitdown_conversion.log' 获取详情。")
```

## 处理异常

在转换过程中处理潜在错误（如不支持的文件格式或损坏的文件）至关重要。以下是批量转换脚本的增强版本，带有强大的异常处理和日志记录。

主要功能：
- 详细的异常层次结构（UnsupportedFormatException、FileConversionException）
- 带时间戳和严重级别的全面日志记录
- 转换前的文件存在性验证
- 针对不同失败场景的清晰错误消息
- 使用适当编码的安全文件处理
- 通过控制台输出跟踪进度
- 将详细日志记录到单独文件以便调试

```python
from markitdown import MarkItDown, FileConversionException, UnsupportedFormatException
from openai import OpenAI  # 可选：用于 LLM 集成
from kaggle_secrets import UserSecretsClient
import os
import logging

# 配置日志
logging.basicConfig(
    filename='/kaggle/working/markitdown_conversion.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 安全获取 API 密钥
user_secrets = UserSecretsClient()

# 初始化 OpenAI 客户端（将 'my-openai-api-key' 替换为您的实际密钥名称）
client = OpenAI(api_key=user_secrets.get_secret("my-openai-api-key"))

# 使用 LLM 集成初始化 MarkItDown
md_converter = MarkItDown(
    llm_client=client,
    llm_model="gpt-4o"  # 指定所需的 LLM 模型
)

# 输入文件所在的目录
input_dir = "/kaggle/input/unstructured-files/"

# 输出目录
output_dir = "/kaggle/working/"

# 输入文件列表
input_files = [
    "data.pdf",
    "data.docx",
    "data.xlsx",
    "data.pptx",
    "data.html",
    "data.txt",
    "data.csv",
    "data.mp3",
    "bar.jpg",
    "tabular.jpg",
    "bubble.png",
    "line.png"
]

for file_name in input_files:
    input_path = os.path.join(input_dir, file_name)
    output_file_name = f"{file_name}.md"
    output_path = os.path.join(output_dir, output_file_name)
    
    try:
        # 检查输入文件是否存在
        if not os.path.isfile(input_path):
            logging.warning(f"未找到文件：{input_path}")
            print(f"警告：未找到文件：{file_name}")
            continue
        
        # 将文件转换为 Markdown
        result = md_converter.convert(input_path)
        
        # 将结果保存到 Markdown 文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.text_content)
        
        logging.info(f"成功将 {file_name} 转换为 {output_file_name}")
        print(f"成功将 {file_name} 转换为 {output_file_name}")
        
    except UnsupportedFormatException as ufe:
        logging.error(f"文件 {file_name} 的格式不受支持：{ufe}")
        print(f"错误：文件 {file_name} 的格式不受支持：{ufe}")
        
    except FileConversionException as fce:
        logging.error(f"文件 {file_name} 转换失败：{fce}")
        print(f"错误：文件 {file_name} 转换失败：{fce}")
        
    except Exception as e:
        logging.error(f"文件 {file_name} 出现意外错误：{e}")
        print(f"错误：文件 {file_name} 出现意外错误：{e}")

print("\n批量转换完成。请查看 'markitdown_conversion.log' 获取详情。")
```

---

## 命令行使用

MarkItDown 提供了一个简单的命令行界面 (CLI)，无需编写代码即可将文件转换为 Markdown。您可以使用简单的命令直接从终端进行转换。

**基本转换：**

```bash
markitdown path-to-file.pdf > document.md
```

**指定输出文件：**

```bash
markitdown path-to-file.pdf -o document.md
```

**管道内容：**

```bash
cat path-to-file.pdf | markitdown
```

**说明**：
- **基本转换**：将 `path-to-file.pdf` 转换为 Markdown 并将输出重定向到 `document.md`。
- **指定输出文件**：使用 `-o` 标志指定输出 Markdown 文件的名称。
- **管道内容**：允许您将文件内容直接通过管道传输到 MarkItDown 进行转换。

---

## 结论
MarkItDown 提供了一个强大而灵活的解决方案，可将多种文件格式转换为 Markdown。无论您处理的是文档、电子表格、演示文稿、媒体文件还是网页内容，MarkItDown 都能简化转换过程，节省您的时间和精力。

**关键要点**：
- **多功能性**：支持多种文件格式，适用于各种使用场景。
- **易用性**：简单的 API 便于集成到 Python 项目中。
- **扩展性**：轻松添加对新格式的支持或集成高级功能，如基于 LLM 的图像描述。
- **批量处理**：通过全面的日志记录和错误处理，高效地同时转换多个文件。

通过将 MarkItDown 融入您的工作流程，您可以提高生产力，保持文档的一致性，并利用 Markdown 的强大功能完成您的项目。无论您是开发人员、数据科学家还是内容创作者，MarkItDown 都是您工具库中不可或缺的利器。