/**
 * Regression: derived history must be wire-legal regardless of surface append
 * order. A notice appended while a tool executes (between the assistant
 * tool-call and its `tool/result`) used to be projected between them and
 * providers rejected the request with
 * `400 messages with role 'tool' must be a response to a preceeding message
 * with 'tool_calls'` (session-aa38d8c0-e197-4e5e-bf1c-55c96fdab089).
 */
import { describe, expect, it } from 'vitest'
import type { AssistantMessage, Message, ToolResultMessage } from '@deepseek-ai/dsh-llm'
import { createMessage, createToolResultMessage, createUserMessage, ToolCallId } from '@deepseek-ai/dsh-llm'
import { repairDerivedHistory, Session, SessionId } from '../src/index.ts'

const modelSource = { kind: 'model', provider: 'mock', model: 'mock' } as const

function assistantWithCalls(...ids: string[]): AssistantMessage {
  return createMessage({
    role: 'assistant',
    source: modelSource,
    content: ids.map(id => ({ type: 'tool-call' as const, id: ToolCallId(id), name: 'echo', arguments: '{}' })),
  })
}

function userNotice(text: string): Message {
  return createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } })
}

function toolResult(id: string, text = 'ok'): ToolResultMessage {
  return createToolResultMessage({ callId: ToolCallId(id), content: [{ type: 'text', text }], isError: false })
}

describe('repairDerivedHistory', () => {
  it('moves a notice that surface order placed inside a tool exchange after the result', () => {
    const call = assistantWithCalls('c1')
    const notice = userNotice('replan')
    const result = toolResult('c1')
    const repaired = repairDerivedHistory([call, notice, result])
    expect(repaired.map(m => m.role)).toEqual(['assistant', 'tool', 'user'])
    // Identity is preserved: same objects, only order changes.
    expect(repaired[0]).toBe(call)
    expect(repaired[1]).toBe(result)
    expect(repaired[2]).toBe(notice)
  })

  it('leaves an already-legal history untouched', () => {
    const history = [userNotice('hi'), assistantWithCalls('c1'), toolResult('c1'), userNotice('next')]
    expect(repairDerivedHistory(history)).toEqual(history)
  })

  it('defers a mid-exchange notice until the exchange fully closes', () => {
    const call = assistantWithCalls('c1', 'c2')
    const notice = userNotice('mid-flight')
    const repaired = repairDerivedHistory([call, notice, toolResult('c1'), toolResult('c2')])
    expect(repaired.map(m => m.role)).toEqual(['assistant', 'tool', 'tool', 'user'])
    expect(repaired[3]).toBe(notice)
  })

  it('leaves an orphan tool result in place when no exchange is open', () => {
    const orphan = toolResult('ghost')
    const history = [orphan, userNotice('u'), assistantWithCalls('c2'), toolResult('c2')]
    const repaired = repairDerivedHistory(history)
    expect(repaired.map(m => m.role)).toEqual(['tool', 'user', 'assistant', 'tool'])
    expect(repaired[0]).toBe(orphan)
  })

  it('passes through empty and single-message histories', () => {
    expect(repairDerivedHistory([])).toEqual([])
    const only = userNotice('solo')
    expect(repairDerivedHistory([only])).toEqual([only])
  })
})

describe('Session.deriveMessages wire order', () => {
  it('returns a tool exchange closed even when a notice was appended mid-call', () => {
    const session = Session.create(SessionId('derived-repair'))
    session.append('turn/start', { turn: 1 })
    session.append('user/message', userNotice('go') as never, { surfaceOp: 'append' })
    session.append('assistant/message', {
      turn: 1, step: 1, stream: [],
      message: assistantWithCalls('c1'),
    }, { surfaceOp: 'append' })
    session.append('tool/call', { turn: 1, step: 1, callId: ToolCallId('c1'), name: 'echo', arguments: '{}' })
    // The mid-exchange notice: appended after the call started, before its result.
    session.append('user/message', userNotice('replan notice') as never, { surfaceOp: 'append' })
    session.append('tool/result', {
      turn: 1, step: 1,
      message: toolResult('c1'),
    }, { surfaceOp: 'append' })
    session.append('turn/end', { turn: 1, reason: { kind: 'completed' } })

    const messages = session.deriveMessages()
    expect(messages.map(m => m.role)).toEqual(['user', 'assistant', 'tool', 'user'])
    expect(messages[2]).toMatchObject({ role: 'tool', toolCallId: 'c1' })
  })
})
