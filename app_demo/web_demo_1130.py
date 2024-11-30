from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/')
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>简单代码补全演示</title>
    <style>
        body {
            font-family: 'Monaco', 'Menlo', monospace;
            margin: 0;
            padding: 20px;
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        #editor-container {
            position: relative;
            width: 800px;
            margin: 0 auto;
        }
        #editor {
            width: 100%;
            height: 400px;
            background-color: #1e1e1e;
            color: #d4d4d4;
            border: 1px solid #454545;
            padding: 10px;
            font-size: 14px;
            line-height: 1.5;
            resize: none;
        }
        #suggestions {
            position: absolute;
            background-color: #252526;
            border: 1px solid #454545;
            padding: 5px;
            display: none;
            z-index: 1000;
        }
        .suggestion-item {
            padding: 5px 10px;
            cursor: pointer;
            color: #d4d4d4;
        }
        .suggestion-item:hover {
            background-color: #37373d;
        }
    </style>
</head>
<body>
    <div id="editor-container">
        <textarea id="editor" spellcheck="false" placeholder="开始输入..."></textarea>
        <div id="suggestions"></div>
    </div>

    <script>
        const editor = document.getElementById('editor');
        const suggestions = document.getElementById('suggestions');
        let typingTimer;

        async function getSuggestions(cursorPos) {
            try {
                const response = await fetch('/complete', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: editor.value,
                        position: cursorPos
                    })
                });
                const data = await response.json();
                return data.suggestions;
            } catch (error) {
                console.error('Error getting suggestions:', error);
                return [];
            }
        }

        function showSuggestions(suggestionText, cursorPos) {
            const rect = editor.getBoundingClientRect();
            const coords = getCaretCoordinates(editor, cursorPos);
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

            suggestions.style.display = 'block';
            suggestions.style.left = (rect.left + coords.left) + 'px';
            suggestions.style.top = (rect.top + coords.top + scrollTop + 20) + 'px';
            suggestions.innerHTML = `
                <div class="suggestion-item" onclick="insertSuggestion('${suggestionText}')">
                    ${suggestionText}
                </div>
            `;
        }

        function getCaretCoordinates(element, position) {
            const clone = element.cloneNode(true);
            clone.style.visibility = 'hidden';
            clone.style.position = 'absolute';
            document.body.appendChild(clone);
            clone.value = element.value.substring(0, position);

            const coordinates = {
                top: clone.scrollHeight,
                left: clone.scrollWidth
            };

            document.body.removeChild(clone);
            return coordinates;
        }

        function insertSuggestion(text) {
            const cursorPos = editor.selectionStart;
            const currentText = editor.value;
            editor.value = currentText.substring(0, cursorPos) + text + currentText.substring(cursorPos);
            editor.focus();
            suggestions.style.display = 'none';
        }

        editor.addEventListener('input', (e) => {
            const cursorPos = e.target.selectionStart;
            clearTimeout(typingTimer);

            if (e.target.value.length > 0) {
                // 使用防抖，避免频繁请求
                typingTimer = setTimeout(async () => {
                    const suggestionList = await getSuggestions(cursorPos);
                    if (suggestionList && suggestionList.length > 0) {
                        showSuggestions(suggestionList[0], cursorPos);
                    }
                }, 300);
            } else {
                suggestions.style.display = 'none';
            }
        });

        editor.addEventListener('blur', () => {
            setTimeout(() => {
                suggestions.style.display = 'none';
            }, 200);
        });

        // 防止点击建议时失去焦点
        suggestions.addEventListener('mousedown', (e) => {
            e.preventDefault();
        });
    </script>
</body>
</html>
    """


@app.route('/complete', methods=['POST'])
def complete():
    return jsonify({'suggestions': ["it is a good day"]})


if __name__ == '__main__':
    app.run(debug=True)