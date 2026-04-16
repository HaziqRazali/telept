import 'package:flutter/material.dart';

import '../config.dart';
import '../services/chat_service.dart';

/// Gemini-powered chatbot panel — standard chat UI.
/// Shows a message bubbles list + text input at the bottom.
class ChatPanel extends StatefulWidget {
  const ChatPanel({super.key});

  @override
  State<ChatPanel> createState() => _ChatPanelState();
}

class _ChatPanelState extends State<ChatPanel> {
  ChatService? _service;
  final _textController = TextEditingController();
  final _scrollController = ScrollController();
  bool _isLoading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initService();
  }

  void _initService() {
    if (AppConfig.hasGeminiKey) {
      _service = ChatService(AppConfig.geminiApiKey);
    }
  }

  @override
  void dispose() {
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _sendMessage() async {
    final text = _textController.text.trim();
    if (text.isEmpty || _isLoading) return;
    if (_service == null) {
      setState(() => _error = 'No Gemini API key set. Go to Settings (⚙) to add one.');
      return;
    }

    _textController.clear();
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      await _service!.send(text);
    } catch (e) {
      setState(() => _error = 'Error: $e');
    } finally {
      setState(() => _isLoading = false);
      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final messages = _service?.messages ?? [];

    return Column(
      children: [
        // ── Header ─────────────────────────────────────────────────────
        Container(
          color: Colors.grey[900],
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Row(
            children: [
              const Icon(Icons.smart_toy_outlined, color: Colors.white70, size: 18),
              const SizedBox(width: 8),
              const Text(
                'Gemini Assistant',
                style: TextStyle(
                  color: Colors.white70,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const Spacer(),
              if (_service != null)
                IconButton(
                  icon: const Icon(Icons.refresh, color: Colors.white38, size: 18),
                  tooltip: 'Clear chat',
                  onPressed: () => setState(() => _service!.clear()),
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                ),
            ],
          ),
        ),

        // ── Messages ───────────────────────────────────────────────────
        Expanded(
          child: messages.isEmpty
              ? _buildEmptyState()
              : ListView.builder(
                  controller: _scrollController,
                  padding: const EdgeInsets.all(8),
                  itemCount: messages.length,
                  itemBuilder: (_, i) => _MessageBubble(msg: messages[i]),
                ),
        ),

        // ── Error banner ───────────────────────────────────────────────
        if (_error != null)
          Container(
            color: Colors.red[900],
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            width: double.infinity,
            child: Text(
              _error!,
              style: const TextStyle(color: Colors.white, fontSize: 11),
            ),
          ),

        // ── Loading indicator ──────────────────────────────────────────
        if (_isLoading)
          const LinearProgressIndicator(minHeight: 2),

        // ── Input bar ──────────────────────────────────────────────────
        Container(
          color: Colors.grey[850],
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _textController,
                  style: const TextStyle(color: Colors.white, fontSize: 13),
                  maxLines: null,
                  textInputAction: TextInputAction.send,
                  onSubmitted: (_) => _sendMessage(),
                  decoration: InputDecoration(
                    hintText: AppConfig.hasGeminiKey
                        ? 'Ask about range of motion...'
                        : 'Set Gemini API key in ⚙ Settings first',
                    hintStyle: const TextStyle(color: Colors.white38, fontSize: 12),
                    filled: true,
                    fillColor: Colors.grey[800],
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(20),
                      borderSide: BorderSide.none,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 6),
              IconButton(
                icon: Icon(
                  Icons.send,
                  color: _isLoading ? Colors.white24 : Colors.blue[300],
                  size: 22,
                ),
                onPressed: _isLoading ? null : _sendMessage,
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.chat_bubble_outline,
                size: 40, color: Colors.white24),
            const SizedBox(height: 12),
            const Text(
              'Ask Gemini about\nrange of motion,\nposture, or exercises.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.white38, fontSize: 12),
            ),
            if (!AppConfig.hasGeminiKey) ...[
              const SizedBox(height: 16),
              const Text(
                'Add your Gemini API key\nin ⚙ Settings to get started.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.orange, fontSize: 11),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _MessageBubble extends StatelessWidget {
  final ChatMessage msg;
  const _MessageBubble({required this.msg});

  @override
  Widget build(BuildContext context) {
    final isUser = msg.isUser;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        mainAxisAlignment:
            isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!isUser) ...[
            CircleAvatar(
              radius: 12,
              backgroundColor: Colors.blue[800],
              child: const Icon(Icons.smart_toy, size: 14, color: Colors.white),
            ),
            const SizedBox(width: 6),
          ],
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: isUser ? Colors.blue[700] : Colors.grey[800],
                borderRadius: BorderRadius.only(
                  topLeft: const Radius.circular(14),
                  topRight: const Radius.circular(14),
                  bottomLeft: Radius.circular(isUser ? 14 : 2),
                  bottomRight: Radius.circular(isUser ? 2 : 14),
                ),
              ),
              child: Text(
                msg.text,
                style: const TextStyle(color: Colors.white, fontSize: 12),
              ),
            ),
          ),
          if (isUser) const SizedBox(width: 6),
        ],
      ),
    );
  }
}
